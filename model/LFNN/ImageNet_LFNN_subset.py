#!/usr/bin/env python
# coding: utf-8
"""ImageNet-subset LFNN training script matching the released 90-epoch configuration."""

# In[1]:


import os
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

print('TensorFlow:', tf.__version__)
tf.keras.utils.set_random_seed(42)


# In[2]:


@dataclass
class LFConfig:
    num_classes: int
    leader_fraction: float = 0.3
    follower_weight: float = 1.0
    leader_weight: float = 1.0
    global_weight: float = 1.0
    use_global_loss: bool = True  # True -> LFNN, False -> LFNN-l
    temperature: float = 1.0


def sparse_ce_per_example(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(
        labels, logits, from_logits=True
    )


class LFWorkerBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        input_dim: int,
        num_workers: int,
        num_classes: int,
        hidden_dim: int = 256,
        name: str = 'lf_block',
    ):
        super().__init__(name=name)
        self.input_proj = tf.keras.layers.Dense(
            hidden_dim, activation='relu', name=f'{name}_proj'
        )
        self.worker_logits = tf.keras.layers.Dense(
            num_workers * num_classes, name=f'{name}_worker_logits'
        )
        self.num_workers = num_workers
        self.num_classes = num_classes

    def call(self, x, training=False):
        h = self.input_proj(x, training=training)
        logits = self.worker_logits(h, training=training)
        logits = tf.reshape(logits, [-1, self.num_workers, self.num_classes])
        return h, logits


# In[3]:


def leader_follower_losses(labels, worker_logits, cfg: LFConfig):
    labels = tf.cast(tf.reshape(labels, [-1]), tf.int32)
    batch_size = tf.shape(worker_logits)[0]
    num_workers = tf.shape(worker_logits)[1]

    flat_logits = tf.reshape(worker_logits, [-1, worker_logits.shape[-1]])
    tiled_labels = tf.repeat(labels, repeats=num_workers)
    per_worker_ce = sparse_ce_per_example(tiled_labels, flat_logits)
    per_worker_ce = tf.reshape(per_worker_ce, [batch_size, num_workers])

    k = tf.maximum(
        1,
        tf.cast(
            tf.math.ceil(cfg.leader_fraction * tf.cast(num_workers, tf.float32)),
            tf.int32,
        ),
    )
    leader_idx = tf.argsort(per_worker_ce, axis=1)[:, :k]
    batch_idx = tf.range(batch_size)[:, None]
    gather_idx = tf.stack([tf.tile(batch_idx, [1, k]), leader_idx], axis=-1)
    leader_logits = tf.gather_nd(worker_logits, gather_idx)  # [B, k, C]
    leader_ce = tf.gather_nd(per_worker_ce, gather_idx)      # [B, k]

    best_pos = tf.argmin(leader_ce, axis=1, output_type=tf.int32)
    best_leader_idx = tf.stack([tf.range(batch_size), best_pos], axis=1)
    best_leader_logits = tf.gather_nd(leader_logits, best_leader_idx)  # [B, C]

    follower_targets = tf.expand_dims(tf.stop_gradient(best_leader_logits), axis=1)
    follower_mse = tf.reduce_mean(
        tf.square(worker_logits - follower_targets), axis=-1
    )  # [B, W]

    leader_mask = tf.reduce_sum(
        tf.one_hot(leader_idx, depth=num_workers, dtype=tf.float32), axis=1
    )  # [B, W]
    follower_mask = 1.0 - leader_mask

    leader_loss = tf.reduce_sum(per_worker_ce * leader_mask) / tf.maximum(
        tf.reduce_sum(leader_mask), 1.0
    )
    follower_loss = tf.reduce_sum(follower_mse * follower_mask) / tf.maximum(
        tf.reduce_sum(follower_mask), 1.0
    )

    worker_probs = tf.nn.softmax(worker_logits, axis=-1)
    best_probs = tf.nn.softmax(best_leader_logits, axis=-1)
    leader_fraction_realized = tf.reduce_mean(
        tf.reduce_sum(leader_mask, axis=1) / tf.cast(num_workers, tf.float32)
    )

    stats = {
        'per_worker_ce': per_worker_ce,
        'leader_idx': leader_idx,
        'leader_mask': leader_mask,
        'follower_mask': follower_mask,
        'best_leader_logits': best_leader_logits,
        'worker_probs': worker_probs,
        'best_probs': best_probs,
        'leader_fraction_realized': leader_fraction_realized,
    }
    return leader_loss, follower_loss, stats


# In[4]:


class LFClassifier(tf.keras.Model):
    def __init__(
        self,
        feature_extractor: tf.keras.Model,
        feature_dim: int,
        num_workers: int,
        cfg: LFConfig,
        hidden_dim: int = 256,
        name: str = 'lf_classifier',
    ):
        super().__init__(name=name)
        self.feature_extractor = feature_extractor
        self.lf_block = LFWorkerBlock(
            feature_dim,
            num_workers=num_workers,
            num_classes=cfg.num_classes,
            hidden_dim=hidden_dim,
        )
        self.global_head = tf.keras.layers.Dense(cfg.num_classes, name='global_head')
        self.cfg = cfg
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.global_loss_tracker = tf.keras.metrics.Mean(name='global_loss')
        self.leader_loss_tracker = tf.keras.metrics.Mean(name='leader_loss')
        self.follower_loss_tracker = tf.keras.metrics.Mean(name='follower_loss')
        self.acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='acc')

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.global_loss_tracker,
            self.leader_loss_tracker,
            self.follower_loss_tracker,
            self.acc_metric,
        ]

    def call(self, x, training=False):
        feats = self.feature_extractor(x, training=training)
        if len(feats.shape) > 2:
            feats = tf.reshape(feats, [tf.shape(feats)[0], -1])
        h, worker_logits = self.lf_block(feats, training=training)
        global_logits = self.global_head(h, training=training)
        return {
            'features': h,
            'worker_logits': worker_logits,
            'global_logits': global_logits,
        }

    def train_step(self, data):
        x, y = data
        y = tf.cast(tf.reshape(y, [-1]), tf.int32)
        with tf.GradientTape() as tape:
            out = self(x, training=True)
            global_logits = out['global_logits']
            worker_logits = out['worker_logits']

            global_loss = tf.reduce_mean(sparse_ce_per_example(y, global_logits))
            leader_loss, follower_loss, stats = leader_follower_losses(
                y, worker_logits, self.cfg
            )

            total_loss = (
                self.cfg.leader_weight * leader_loss
                + self.cfg.follower_weight * follower_loss
            )
            if self.cfg.use_global_loss:
                total_loss = total_loss + self.cfg.global_weight * global_loss
            total_loss += tf.add_n(self.losses) if self.losses else 0.0

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.loss_tracker.update_state(total_loss)
        self.global_loss_tracker.update_state(global_loss)
        self.leader_loss_tracker.update_state(leader_loss)
        self.follower_loss_tracker.update_state(follower_loss)
        self.acc_metric.update_state(y, tf.nn.softmax(global_logits, axis=-1))

        return {m.name: m.result() for m in self.metrics} | {
            'leader_fraction': stats['leader_fraction_realized']
        }

    def test_step(self, data):
        x, y = data
        y = tf.cast(tf.reshape(y, [-1]), tf.int32)

        out = self(x, training=False)
        global_logits = out['global_logits']
        worker_logits = out['worker_logits']

        global_loss = tf.reduce_mean(sparse_ce_per_example(y, global_logits))
        leader_loss, follower_loss, stats = leader_follower_losses(
            y, worker_logits, self.cfg
        )

        total_loss = (
            self.cfg.leader_weight * leader_loss
            + self.cfg.follower_weight * follower_loss
        )
        if self.cfg.use_global_loss:
            total_loss = total_loss + self.cfg.global_weight * global_loss

        self.loss_tracker.update_state(total_loss)
        self.global_loss_tracker.update_state(global_loss)
        self.leader_loss_tracker.update_state(leader_loss)
        self.follower_loss_tracker.update_state(follower_loss)
        self.acc_metric.update_state(y, tf.nn.softmax(global_logits, axis=-1))

        return {m.name: m.result() for m in self.metrics} | {
            'leader_fraction': stats['leader_fraction_realized']
        }


# In[5]:


FULL_IMAGENET = False


@dataclass
class Config(LFConfig):
    image_shape: Tuple[int, int, int] = (224, 224, 3)
    batch_size: int = 64
    epochs: int = 90
    learning_rate: float = 1e-3
    num_workers: int = 32
    hidden_dim: int = 512
    train_samples: int = 3200
    val_samples: int = 2000


# Authoritative configuration for model/LFNN-BPfree/
# imagenet__lfnn_l_training_log.txt. With batch_size=64 and
# train_samples=3200, each epoch has the recorded 50 training steps.
CFG = Config(num_classes=1000, leader_fraction=0.9, use_global_loss=True)


def preprocess(example, image_size=(224, 224)):
    x = tf.image.resize(tf.cast(example['image'], tf.float32) / 255.0, image_size)
    y = tf.cast(example['label'], tf.int32)
    return x, y


def load_datasets(cfg: Config, full_imagenet: bool = False):
    dataset_name = 'imagenet2012' if full_imagenet else 'imagenet_resized/64x64'
    train_split = 'train'
    val_split = 'validation'

    train_ds = tfds.load(dataset_name, split=train_split, shuffle_files=True)
    val_ds = tfds.load(dataset_name, split=val_split, shuffle_files=False)

    if not full_imagenet:
        train_ds = train_ds.take(cfg.train_samples)
        val_ds = val_ds.take(cfg.val_samples)

    train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(4096).batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds


def build_feature_extractor(cfg: Config):
    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=None,
        input_shape=cfg.image_shape,
        pooling='avg',
    )
    inp = tf.keras.Input(shape=cfg.image_shape)
    x = backbone(inp)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    return tf.keras.Model(inp, x, name='imagenet_backbone')


def build_model(cfg: Config):
    backbone = build_feature_extractor(cfg)
    return LFClassifier(
        feature_extractor=backbone,
        feature_dim=512,
        num_workers=cfg.num_workers,
        cfg=cfg,
        hidden_dim=cfg.hidden_dim,
        name='imagenet_lfnn',
    )


# In[6]:


train_ds, val_ds = load_datasets(CFG, full_imagenet=FULL_IMAGENET)
print(train_ds, val_ds)


# In[7]:


model = build_model(CFG)
model.compile(optimizer=tf.keras.optimizers.Adam(CFG.learning_rate))
model.summary()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=CFG.epochs,
)

results = model.evaluate(val_ds, return_dict=True)
print(results)
