## Learning Objectives

By the end of this lesson, you will:

**Understand:**
- How convolution operates as a parameter-sharing pattern detector that encodes translation equivariance and locality, and why this makes CNNs dramatically more efficient than fully connected networks for spatial data
- How the feature hierarchy (edges to textures to parts to objects) emerges naturally from training, and how residual connections solved the degradation problem that limited network depth
- How Vision Transformers (ViTs) differ from CNNs in their inductive biases, and when each architecture is the better choice

**Apply:**
- Use pretrained CNN models (ResNet, EfficientNet) for transfer learning with feature extraction, fine-tuning, and progressive unfreezing strategies
- Calculate output dimensions for convolutional layers given kernel size, stride, padding, and dilation

**Analyze:**
- Evaluate the tradeoffs between CNNs and ViTs for a given deployment scenario, considering data availability, compute constraints, latency requirements, and accuracy targets

## Prerequisites

- **Layers and activations** -- CNNs build on the concept of stacking layers with non-linear activations to learn hierarchical features (see [Fundamentals](../fundamentals/COURSE.md))
- **Training loops and backpropagation** -- you need to understand training mechanics to train and fine-tune CNNs effectively (see [Training Mechanics](../training-mechanics/COURSE.md))

---

# Convolutional Neural Networks (CNNs)

## Why This Lesson Matters

CNNs are the architecture that launched the deep learning revolution. While transformers now dominate vision tasks at scale, CNNs remain the foundation of how we think about spatial feature learning, and they are still the practical choice for production vision systems on edge devices, mobile apps, and latency-sensitive workloads. Understanding CNNs deeply gives you the conceptual vocabulary for all of computer vision.

---

## 1. What Convolution Is

### The Core Operation

A convolution slides a small matrix (the **kernel** or **filter**) across the input, computing a dot product at each position. The output is a **feature map** — a spatial map of where the kernel's pattern was detected.

```
Input (5x5):          Kernel (3x3):        Output (3x3):
1 0 1 0 1             1 0 1                ?  ?  ?
0 1 0 1 0             0 1 0                ?  ?  ?
1 0 1 0 1             1 0 1                ?  ?  ?
0 1 0 1 0
1 0 1 0 1

Top-left output = sum of elementwise multiply of kernel with top-left 3x3 patch
= 1*1 + 0*0 + 1*1 + 0*0 + 1*1 + 0*0 + 1*1 + 0*0 + 1*1 = 5
```

The kernel acts as a **pattern detector**. A horizontal edge detector kernel:

```
 1  1  1
 0  0  0
-1 -1 -1
```

This gives a high response wherever there is a bright-above-dark transition — a horizontal edge. A vertical edge detector rotates this 90 degrees.

### Why Convolution and Not Just Fully Connected Layers?

Consider processing a 224x224 RGB image with a fully connected layer:
- Input: 224 * 224 * 3 = 150,528 neurons
- Hidden layer of 256 neurons: 150,528 * 256 = **38.5 million parameters** in one layer
- This is massive, will overfit immediately on small datasets, and ignores spatial structure

A convolutional layer with 64 filters of size 3x3:
- Parameters: 64 * 3 * 3 * 3 = **1,728 parameters**
- 22,000x fewer parameters, and each one is reused across the entire image

### Parameter Sharing — The Key Insight

**The same kernel is used at every spatial position.** A 3x3 kernel has only 9 parameters, but it is applied thousands of times across the image.

This encodes the assumption that **a useful feature in one part of the image is useful in other parts too.** An edge is an edge regardless of where it appears. A texture is a texture whether it is in the top-left or bottom-right.

This assumption is called **translation equivariance** — if the input shifts, the output shifts by the same amount, but the detected features remain the same.

---

## 2. Why CNNs Work for Images

CNNs encode two critical **inductive biases** about images:

### Translation Equivariance

Because the same kernel is applied everywhere, a cat detected in the top-left produces the same activation pattern as a cat in the bottom-right. The network does not need to learn "cat at position (10, 10)" and "cat at position (200, 200)" separately.

Technically, convolutions are **translation equivariant**: shifting the input shifts the output by the same amount. The final classification becomes **translation invariant** through pooling and global average pooling — the model's prediction does not change if the object moves.

### Locality

Each neuron in a conv layer only "sees" a small spatial region of the input (its **receptive field**). Early layers see 3x3 patches. Stacking layers increases the receptive field:

```
Layer 1: 3x3 receptive field   (edges, textures)
Layer 2: 5x5 receptive field   (corners, simple patterns)
Layer 3: 7x7 receptive field   (parts, shapes)
Layer 4: 11x11 receptive field (larger structures)
...
Deep layers: large receptive field (objects, scenes)
```

This mirrors how the visual cortex works — simple cells detect edges, complex cells detect shapes, and higher areas detect objects. The hierarchy is not programmed; it emerges from training.

---

## 3. Kernels and Filters: What They Learn

### The Hierarchy of Learned Features

This has been empirically verified by visualizing learned filters at each layer of trained CNNs:

```
FEATURE HIERARCHY:

Layer 1:  Oriented edges (horizontal, vertical, diagonal)
          Color blobs (red patches, blue gradients)
          Frequency patterns (Gabor-like filters)

Layer 2:  Textures (stripes, dots, grids)
          Corners and T-junctions
          Color combinations

Layer 3:  Parts of objects (wheels, eyes, windows, fur)
          Repeating textures at larger scales

Layer 4+: Entire objects (faces, cars, animals)
          Scene-level features (indoor vs outdoor)
          Class-specific patterns
```

### Channels and Depth

A single convolutional layer applies **multiple** filters to produce multiple output channels:

```
Input:   (H, W, C_in)    e.g., (224, 224, 3) for RGB
Filters: (K, K, C_in) x C_out   e.g., 64 filters of size (3, 3, 3)
Output:  (H', W', C_out)  e.g., (224, 224, 64)

Each filter produces one output channel.
64 filters = 64 different feature detectors running in parallel.
```

As you go deeper, the number of channels typically increases (from 3 to 64 to 128 to 256 to 512) while the spatial dimensions decrease (from 224 to 112 to 56 to 28 to 14 to 7). The representation transitions from "large spatial map with few features" to "small spatial map with rich features."

### 1x1 Convolutions — Channel Mixing

A 1x1 convolution does not look at spatial neighborhoods at all. It takes the vector of channels at each spatial position and projects it to a new channel dimension:

```
Input:  (H, W, 256)
1x1 conv with 64 filters: (H, W, 64)
```

This is equivalent to a fully connected layer applied independently to each spatial position. Uses:
- **Dimensionality reduction** (bottleneck layers in ResNet, Inception)
- **Channel mixing** (combining information across feature maps)
- **Increasing non-linearity** (adding another activation after the 1x1 conv)

---

## 4. Key Hyperparameters

| Hyperparameter | What It Controls | Typical Values |
|----------------|-----------------|----------------|
| **Kernel size** | Spatial extent of the filter | 3x3 (standard), 1x1 (channel mixing), 7x7 (first layer) |
| **Stride** | Step size of the sliding window | 1 (preserve size) or 2 (halve dimensions) |
| **Padding** | Zeros added around borders | `same` (preserve size) or `valid` (shrink) |
| **Dilation** | Gaps between kernel elements | 1 (standard) or 2+ (larger receptive field) |
| **Groups** | Split channels into independent groups | 1 (standard) or C_in (depthwise) |

### Output Size Formula

```
output_size = floor((input_size + 2*padding - dilation*(kernel_size-1) - 1) / stride) + 1
```

Simplified for the common case (dilation=1):
```
output_size = floor((input_size + 2*padding - kernel_size) / stride) + 1
```

**Memorize this.** It comes up in every CNN design and interview. Common configurations:
- 3x3 kernel, stride 1, padding 1: output = input (preserves spatial size)
- 3x3 kernel, stride 2, padding 1: output = input/2 (downsamples by 2x)
- 7x7 kernel, stride 2, padding 3: output = input/2 (common first layer)

### Depthwise Separable Convolutions

Standard convolution: each filter spans all input channels. Cost: K*K*C_in*C_out.

Depthwise separable convolution splits this into two steps:
1. **Depthwise:** One K*K filter per input channel (groups=C_in). Cost: K*K*C_in
2. **Pointwise:** 1x1 convolution to mix channels. Cost: C_in*C_out

Total cost: K*K*C_in + C_in*C_out vs K*K*C_in*C_out. For 3x3 filters, this is roughly **9x cheaper.** Used in MobileNet, EfficientNet, and anywhere efficiency matters.

---

### Check Your Understanding

1. An input feature map is 64x64 with 128 channels. You apply a convolutional layer with 256 filters of size 3x3, stride 2, and padding 1. What is the output shape, and how many learnable parameters does this layer have (including bias)?
2. Why do CNNs use small 3x3 kernels stacked in multiple layers rather than a single large 7x7 kernel? Give both the parameter efficiency argument and the representational power argument.
3. A common misconception is that a 1x1 convolution does nothing useful because it has no spatial extent. What does a 1x1 convolution actually compute, and why is it important?

<details>
<summary>Answers</summary>

1. Output spatial size = floor((64 + 2*1 - 3) / 2) + 1 = floor(63/2) + 1 = 32. Output shape: (32, 32, 256). Parameters: 256 filters * (3 * 3 * 128 + 1 bias) = 256 * 1153 = 295,168 parameters.
2. Parameter efficiency: Two 3x3 layers have 2 * 9 = 18 parameters per input-output channel pair vs 49 for a 7x7 layer, a 2.7x saving. Representational power: Two 3x3 layers have the same 5x5 receptive field as one 5x5 kernel, but include two non-linear activations instead of one, enabling the network to learn more complex functions within the same receptive field.
3. A 1x1 convolution computes a learned linear combination of channels at each spatial position independently. It is equivalent to a fully connected layer applied at every pixel. Uses include: dimensionality reduction (bottleneck layers in ResNet, Inception), channel mixing (combining information across feature maps), and adding non-linearity (when followed by an activation). It is a key building block in modern architectures.

</details>

---

## 5. Pooling

### Max Pooling

Takes the maximum value in each spatial window:

```
Input (4x4):     Max Pool 2x2, stride 2:     Output (2x2):
1 3 2 1          max(1,3,5,2)=5               5  3
5 2 1 3          max(2,1,1,3)=3               7  4
4 7 3 1          max(4,7,3,1)=7
2 1 4 2          max(3,1,4,2)=4
```

### Why Downsample?

1. **Reduces computation:** Each 2x2 pooling halves spatial dimensions, quartering the number of activations
2. **Increases receptive field:** After pooling, each neuron in the next conv layer effectively covers a larger input region
3. **Provides local translation invariance:** A feature slightly shifted within the pooling window still produces the same max value

### Average Pooling

Takes the mean instead of the max. Less aggressive than max pooling (preserves more information about the average response rather than the strongest response).

### Global Average Pooling (GAP) — The Modern Standard

Instead of flattening the final feature map and using a huge linear layer, GAP averages each channel into a single number:

```python
# Final conv output: (batch, 512, 7, 7)
# Global Average Pooling: (batch, 512)  — one number per channel
# Then a single linear layer: (batch, num_classes)
```

GAP replaced the fully connected "classifier head" of early CNNs (which had millions of parameters), dramatically reducing overfitting. It also makes the model agnostic to input resolution — any spatial size can be averaged.

### Modern Trend: Strided Convolutions

Many modern architectures (ResNet, EfficientNet) replace max pooling with **strided convolutions** (stride=2). This lets the network **learn** the downsampling operation rather than hardcoding max or average. The network decides what information to keep.

---

## 6. Key Architectures: The Evolution

Understanding how CNN architectures evolved reveals the key insights that shaped all of deep learning.

### LeNet-5 (1998, Yann LeCun)

The original CNN. Built for digit recognition (MNIST). 5 layers, ~60K parameters.

```
Input(32x32) -> Conv(5x5) -> Pool -> Conv(5x5) -> Pool -> FC(120) -> FC(84) -> Output(10)
```

**Why it matters:** Proved that learned features outperform hand-engineered ones. Introduced the conv-pool-conv-pool-FC pattern that dominated for 15 years. Deployed commercially at AT&T for check reading.

### AlexNet (2012, Krizhevsky et al.)

**The ImageNet moment.** Won the 2012 ImageNet competition by a massive margin (15.3% vs 26.2% top-5 error). 8 layers, ~60M parameters. This single paper launched the deep learning revolution.

**Key innovations:**
- ReLU activation (replacing sigmoid/tanh — this alone was a breakthrough for deep training)
- Dropout for regularization (randomly zeroing neurons during training)
- Data augmentation (random crops, horizontal flips, color jittering)
- GPU training (split across two GTX 580s with 3GB each)
- Local response normalization (later replaced by batch normalization)

**Why it matters:** Before AlexNet, computer vision was dominated by hand-crafted features (SIFT, HOG, Haar). After AlexNet, everything was neural. This was the moment the field shifted.

### VGGNet (2014, Simonyan & Zisserman)

Showed that **depth matters** more than clever architecture. 16-19 layers, using only 3x3 convolutions stacked uniformly.

**Key insight:** Two 3x3 conv layers have the same receptive field as one 5x5, but fewer parameters (2 * 3^2 = 18 vs 5^2 = 25) and more non-linearity (two ReLUs vs one). Three 3x3 layers = one 7x7, with even greater savings.

```
VGG-16 Architecture:
[3x3 conv, 64] x 2 -> pool
[3x3 conv, 128] x 2 -> pool
[3x3 conv, 256] x 3 -> pool
[3x3 conv, 512] x 3 -> pool
[3x3 conv, 512] x 3 -> pool
FC(4096) -> FC(4096) -> FC(1000)
```

**Why it matters:** Established the principle "go deeper with small filters." VGG-16 is still used as a feature extractor and style transfer backbone because its features are clean and hierarchical.

**Problem:** 138M parameters. The fully connected layers alone have ~120M parameters. Enormously expensive in memory.

### GoogLeNet / Inception (2014, Szegedy et al.)

Introduced the **Inception module** — apply multiple filter sizes (1x1, 3x3, 5x5) in parallel and concatenate the results. Let the network decide which scale is useful at each layer.

```
                Input
         /    |    |     \
      1x1   1x1   1x1   3x3 max pool
       |     |     |        |
      ...  3x3   5x5      1x1
       \    |     |       /
         Concatenate
```

**Key innovations:**
- 1x1 convolutions for dimensionality reduction (bottleneck layers) before expensive 3x3 and 5x5 convolutions
- Auxiliary classifiers at intermediate layers (injected gradient signal deeper into the network)
- Global average pooling instead of FC layers — dropped from 138M to 7M parameters

### ResNet (2015, He et al.) — The Game Changer

The single most important CNN architecture. Introduced **residual connections** (skip connections). Covered in depth in the next section.

### DenseNet (2017, Huang et al.)

Instead of adding skip connections (ResNet), **concatenate** outputs from all previous layers:

```
x0 -> [layer1] -> x1
[x0, x1] -> [layer2] -> x2
[x0, x1, x2] -> [layer3] -> x3
```

Every layer receives all previous feature maps as input. This encourages feature reuse and requires fewer parameters per layer.

### EfficientNet (2019, Tan & Le)

Used **neural architecture search (NAS)** to find the optimal balance of depth, width, and resolution. Introduced **compound scaling** — scale all three dimensions together with a fixed ratio.

```
depth   *= alpha^phi
width   *= beta^phi
resolution *= gamma^phi
where alpha * beta^2 * gamma^2 = 2  (roughly doubles FLOPS)
```

Key building block: **MBConv** (Mobile Inverted Bottleneck):
1. Expand channels with 1x1 conv
2. Depthwise 3x3 conv
3. Squeeze-and-excitation (SE) for channel attention
4. Compress back with 1x1 conv
5. Skip connection

**Why it matters:** Achieved state-of-the-art accuracy with 8x fewer parameters than previous models. Proved that architecture design can be automated and that the balance between dimensions matters more than any single dimension.

### ConvNeXt (2022, Liu et al.)

"What if we took a ResNet and modernized it with every trick from transformers?" Result: a pure CNN that matches Vision Transformers on ImageNet.

Changes from standard ResNet:
- Larger kernel (7x7 depthwise conv instead of 3x3)
- Fewer activation functions (only one per block, not after every conv)
- LayerNorm instead of BatchNorm
- GELU activation instead of ReLU
- Inverted bottleneck (wide middle, narrow ends, like transformers)
- Fewer, wider layers

**Key takeaway:** Much of ViT's advantage was the training recipe (data augmentation, regularization, longer training), not the architecture itself. A well-modernized CNN can match ViT.

---

## 7. Residual Connections (Skip Connections): Why ResNet Was Revolutionary

### The Degradation Problem

Before ResNet, a counterintuitive problem plagued deep networks: **adding more layers made performance worse**, even on the *training* set. This was not overfitting — a 56-layer network had higher training error than a 20-layer one.

Why? The 56-layer network should be able to learn the identity function for the extra 36 layers and match the 20-layer network at worst. But gradient-based optimization could not find this solution. The identity function, while simple to write mathematically, was hard to learn through iterated non-linear transformations.

### The Skip Connection Solution

```python
# Standard block:
output = F(x)            # Must learn the full transformation

# Residual block:
output = F(x) + x        # Only learn the RESIDUAL (deviation from identity)
```

Now the layers only need to learn the **residual** `F(x) = desired_output - x`. If the optimal transformation is close to identity (which it often is in deep networks), learning a near-zero residual is much easier than learning an identity mapping from scratch.

**Analogy:** Instead of writing a new report from scratch, you edit the previous draft. The edits (residual) are typically much simpler than the full report.

### The Gradient Highway

During backpropagation, the gradient at layer l is:

```
d_loss/d_x_l = d_loss/d_x_L * (1 + d_F/d_x_l)
```

The **1** term means gradients flow directly from the loss to any layer, undiminished. Even if `d_F/d_x_l` vanishes (the layer's own gradient dies), the gradient is at least `d_loss/d_x_L` through the identity path.

This is why you can train **1000+ layer** ResNets. The gradient has a clear highway from the loss to the earliest layers.

### Bottleneck Design

For deeper models (ResNet-50 and beyond), bottleneck blocks reduce computation:

```
1x1 conv (reduce channels: 256 -> 64)     # Compress
3x3 conv (process at reduced channels: 64 -> 64)  # Compute
1x1 conv (expand back: 64 -> 256)          # Expand
+ skip connection from input
```

This is 3.7x fewer FLOPs than two 3x3 convolutions at 256 channels, with no loss in quality.

### Why Every Modern Architecture Uses Skip Connections

```
ResNet (2015):          output = F(x) + x
Transformers (2017):    output = Attention(x) + x, then FFN(x) + x
DenseNet (2017):        output = [x, F(x)]  (concatenation variant)
U-Net (segmentation):   output = Decoder(x) + Encoder_features
Diffusion models:       Skip connections throughout the U-Net
```

The principle `output = F(x) + x` is the most impactful architectural idea in deep learning. It appears everywhere because it solves the fundamental problem of training deep networks.

---

## 8. Transfer Learning with CNNs

### Why Pretrained Features Work

The early layers of any CNN trained on natural images learn the same features — edges, textures, colors. These are **universal visual features**. Only the deep layers are task-specific.

This means you can take a CNN pretrained on ImageNet (1.2M images, 1000 classes) and adapt it to your task with very little data.

### Strategy 1: Feature Extraction (Small Dataset, <1000 images per class)

```python
model = torchvision.models.resnet50(weights='IMAGENET1K_V2')
for param in model.parameters():
    param.requires_grad = False          # Freeze everything
model.fc = nn.Linear(2048, num_classes)  # Replace final layer
# Train only the new layer — fast, needs minimal data
```

### Strategy 2: Fine-Tuning (Medium Dataset, 1K-100K images per class)

```python
model = torchvision.models.resnet50(weights='IMAGENET1K_V2')
model.fc = nn.Linear(2048, num_classes)

# Differential learning rates: lower for pretrained layers, higher for new layers
optimizer = optim.Adam([
    {'params': model.fc.parameters(), 'lr': 1e-3},       # New head: high LR
    {'params': model.layer4.parameters(), 'lr': 1e-4},    # Late layers: medium
    {'params': model.layer3.parameters(), 'lr': 1e-5},    # Earlier: low
    # Even earlier layers can stay frozen
])
```

### Strategy 3: Progressive Unfreezing

1. Train only the new head for a few epochs
2. Unfreeze the last block, fine-tune with low LR
3. Unfreeze more blocks gradually
4. This prevents **catastrophic forgetting** — early layers' useful features getting overwritten

### When Transfer Learning Fails

- **Domain gap too large:** Medical images, satellite imagery, microscopy look nothing like ImageNet. You may need domain-specific pretraining (e.g., models pretrained on medical data).
- **Task is fundamentally different:** Counting, detection, segmentation may need architectural changes beyond swapping the head.
- **Data distribution shift:** If your production data looks different from training data, no amount of ImageNet pretraining helps.

---

### Check Your Understanding

1. You have a dataset of 500 medical X-ray images across 5 disease categories. Would you use feature extraction or fine-tuning with a pretrained ImageNet model? Justify your choice and mention a potential concern.
2. Explain the difference between translation equivariance and translation invariance. Which one does a convolutional layer provide, and how does the overall CNN classification pipeline achieve the other?
3. Why did ResNet's skip connections enable training networks hundreds of layers deep, when previous architectures degraded in performance beyond 20-30 layers?

<details>
<summary>Answers</summary>

1. Use feature extraction (freeze pretrained backbone, train only the new classification head). With only 500 images (100 per class), fine-tuning the full network would overfit severely. The concern is the domain gap: medical X-rays look very different from ImageNet natural images, so the early layer features (edges, textures) may be less transferable. If performance is poor, consider finding a model pretrained on medical imaging data, or using progressive unfreezing with very low learning rates.
2. A convolutional layer provides translation equivariance: if the input shifts spatially, the output feature map shifts by the same amount. The same features are detected regardless of position. Translation invariance (output unchanged by input shift) is achieved through pooling and global average pooling at the end of the network, which aggregate spatial information into a position-independent representation for classification.
3. Before ResNet, deeper networks suffered from the degradation problem: optimization could not find good solutions because learning an identity mapping through iterated non-linear transformations was difficult. Skip connections solve this by letting layers learn only the residual F(x) = desired_output - x. If the optimal transformation is near-identity (common in deep networks), learning a near-zero residual is much easier. Additionally, the gradient flows through the identity path undiminished (gradient of addition is 1), providing a gradient highway to early layers.

</details>

---

## 9. Vision Transformers (ViT): How Transformers Replaced CNNs

### The Core Idea

Instead of convolutions, treat an image as a **sequence of patches** and process it with a standard transformer:

```
224x224 image -> split into 16x16 patches -> 196 patches (14x14 grid)
Each 16x16x3 patch = 768-dimensional vector (flatten and project)
Prepend a [CLS] token -> 197 tokens
Process with standard transformer encoder
[CLS] output -> classification head
```

### Why It Works

Self-attention computes relationships between all patches — global receptive field from layer 1. No need to stack layers to build up receptive field like CNNs. The model can learn that a distant patch (sky) is relevant to classifying the current patch (bird) from the very first layer.

### Why ViT Needs More Data

CNNs bake in strong inductive biases: locality, translation equivariance, parameter sharing. These act as powerful regularization on small datasets.

ViT has **no spatial inductive biases** — it must learn that nearby patches are related, that features should be translation-invariant, etc. This requires more data:
- ViT trained on ImageNet-1K (1.2M images): worse than ResNet
- ViT trained on ImageNet-21K (14M images): matches ResNet
- ViT trained on JFT-300M (300M images): crushes everything

### DINOv2, SigLIP, CLIP — The Modern Vision Backbone

In 2024-2026, the dominant approach is:
1. Pretrain a ViT on massive data with self-supervised (DINO) or contrastive (CLIP, SigLIP) objectives
2. Use the frozen ViT as a feature extractor
3. Fine-tune a lightweight head for your task

These models produce rich, general-purpose visual representations that transfer to almost any vision task.

---

## 10. When to Use CNNs vs ViTs in 2026

### Use CNNs When:

| Scenario | Why CNN | Recommended Architecture |
|----------|---------|-------------------------|
| Edge/mobile deployment | Smaller, faster, optimized hardware support | MobileNet, EfficientNet-Lite |
| Limited data (<100K images) | Stronger inductive biases regularize better | ResNet-50 with transfer learning |
| Real-time latency requirements | Faster inference at small scale | MobileNetV3, EfficientNet-B0 |
| Embedded systems | Lower memory footprint | Quantized MobileNet |

### Use ViTs When:

| Scenario | Why ViT | Recommended Architecture |
|----------|---------|-------------------------|
| Large-scale pretraining available | Superior performance with enough data | ViT-L with DINOv2 weights |
| Multi-modal tasks | Natural interface with text transformers | CLIP, SigLIP backbone |
| Variable resolution inputs | Handle different sizes via sequence length | ViT with interpolated position embeddings |
| Research / pushing accuracy | Best absolute performance | ViT-Huge or larger |

### The Practical Decision Tree

```
Do you have a pretrained ViT/CLIP model for your domain?
  YES -> Use it. Fine-tune the head.
  NO ->
    Do you have >1M training images?
      YES -> Train or fine-tune a ViT
      NO ->
        Do you need mobile/edge deployment?
          YES -> EfficientNet / MobileNet
          NO -> ResNet-50 with ImageNet pretrained weights
```

### Hybrid Architectures

The boundary is blurring. Many state-of-the-art models combine both:
- **Early CNN layers + later transformer layers** (locality early, global attention late)
- **CNN feature extractor + transformer decoder** (detection and segmentation)
- **Convolutional position encoding in ViTs** (adding locality bias to transformers)

---

## 11. Practical CNN Design Patterns

### The Standard Recipe

```
1. Start with a pretrained model (ResNet-50, EfficientNet-B0, or ViT)
2. Replace the classification head
3. Freeze backbone, train head for 5-10 epochs
4. Unfreeze backbone, fine-tune with 10x lower learning rate
5. Use standard augmentation: RandomResizedCrop, HorizontalFlip, ColorJitter
6. Use AdamW optimizer, cosine LR schedule
7. Train for 30-100 epochs with early stopping on validation loss
```

### Data Augmentation — Free Performance

Data augmentation is the single highest-ROI technique in computer vision. Common augmentations:

| Augmentation | What It Does | When to Use |
|-------------|-------------|------------|
| RandomResizedCrop | Random crop and resize | Always |
| HorizontalFlip | Mirror image | When objects have no left-right semantics |
| ColorJitter | Random brightness, contrast, saturation | Almost always |
| RandomRotation | Rotate by random angle | When orientation is not meaningful |
| MixUp | Blend two images and their labels | Regularization for larger models |
| CutMix | Replace a patch with a patch from another image | Similar to MixUp, often better |
| RandAugment | Random policy of many augmentations | Modern default |
| AugMax | Adversarial augmentation | For robustness |

### Common Debugging Patterns

| Problem | What to Check |
|---------|---------------|
| Model predicts all one class | Class imbalance, learning rate too high, data loading bug |
| Training loss oscillates wildly | Learning rate too high, batch size too small |
| Val accuracy plateaus at random chance | Labels are wrong, data pipeline is broken, model is too simple |
| Good val accuracy, bad test accuracy | Data leakage in val set, or distribution shift in test set |

---

## Common Pitfalls

**Pitfall 1: Training a CNN from scratch when a pretrained model is available**
- Symptom: Poor accuracy after long training, especially on small datasets (<100K images)
- Why: Training from scratch requires learning universal features (edges, textures) that pretrained models already know. On small datasets, the model overfits before learning useful representations
- Fix: Always start with a pretrained model (ResNet-50, EfficientNet) and fine-tune. Training from scratch is only justified with millions of images in a domain far from ImageNet

**Pitfall 2: Using the same learning rate for pretrained and new layers during fine-tuning**
- Symptom: Either the pretrained features get destroyed (LR too high) or the new head learns too slowly (LR too low)
- Why: Pretrained layers contain useful features that need only small adjustments, while randomly initialized new layers need large updates
- Fix: Use differential learning rates -- 1e-3 for the new head, 1e-4 for late pretrained layers, 1e-5 or frozen for early layers

**Pitfall 3: Forgetting to resize or normalize images to match the pretrained model's expectations**
- Symptom: Pretrained model produces random or very poor predictions despite correct architecture setup
- Why: ImageNet-pretrained models expect specific input sizes (224x224) and normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]). Mismatched preprocessing makes the learned features meaningless
- Fix: Always use the same preprocessing as the pretrained model. In torchvision, use the model's built-in transforms or the documented normalization values

**Pitfall 4: Applying horizontal flip augmentation when left-right matters**
- Symptom: Subtle accuracy drop that is hard to diagnose; model confuses left-right properties
- Why: For tasks like reading text in images, medical imaging (heart is on the left), or directional tasks, horizontal flipping creates invalid training examples
- Fix: Only use augmentations that preserve the semantics of your task. Think about which transformations produce valid examples

## Hands-On Exercises

### Exercise 1: Transfer Learning Pipeline
**Goal:** Implement and compare the three transfer learning strategies on a real dataset
**Task:**
1. Download a small image dataset (e.g., Oxford Flowers-102 or Stanford Dogs, ~8K images)
2. Implement three approaches using a pretrained ResNet-50: (a) feature extraction (freeze all, train new head), (b) full fine-tuning (same LR for all layers), (c) differential learning rates (high for head, low for backbone)
3. Train each for 20 epochs, logging training and validation accuracy
4. Compare: final accuracy, training time, and the train-val gap for each approach
**Verify:** Feature extraction should converge fastest but may have lower final accuracy. Full fine-tuning with a single high LR should show overfitting. Differential LR should achieve the best final validation accuracy.

### Exercise 2: Build and Visualize a CNN
**Goal:** Understand what CNN layers learn by building a simple CNN and visualizing its filters
**Task:**
1. Build a 4-layer CNN from scratch in PyTorch for CIFAR-10 (no pretrained weights)
2. Train it to at least 75% accuracy
3. After training, visualize: (a) the learned 3x3 filters of the first convolutional layer (should look like edge detectors), (b) the feature maps (activations) at each layer for a sample image, (c) the receptive field size at each layer
4. Bonus: Apply Grad-CAM to see which regions of the input the model focuses on for its predictions
**Verify:** First-layer filters should show oriented edges and color blobs. Deeper feature maps should be spatially smaller but respond to more complex patterns. Grad-CAM should highlight the object being classified.

---

## 12. Interview Questions

### Conceptual

1. **Why do CNNs use small (3x3) filters instead of large ones?** Two stacked 3x3 layers have the same receptive field as one 5x5 layer but with fewer parameters (18 vs 25) and two non-linearities instead of one, giving more representational power per parameter.

2. **Explain translation equivariance vs translation invariance.** Equivariance means the output shifts when the input shifts (same features detected, just moved). Invariance means the output stays the same regardless of where the feature is. Convolutions are equivariant; the classification output is invariant (achieved through pooling/GAP).

3. **Why did ResNet's skip connections change everything?** They solved the degradation problem where deeper networks paradoxically performed worse even on training data. Skip connections let layers learn residuals (small corrections) rather than full transformations, and provide a gradient highway where the gradient flows through the identity path undiminished. This enabled training networks hundreds of layers deep.

4. **What are depthwise separable convolutions and why do they matter?** They factorize a standard convolution into a depthwise convolution (one filter per channel, spatial only) and a pointwise convolution (1x1, channel mixing only). This reduces computation by roughly K^2 times (9x for 3x3 filters) with minimal accuracy loss. This is why models like MobileNet and EfficientNet can run on phones.

5. **When would you choose a CNN over a Vision Transformer in 2026?** For edge/mobile deployment (CNNs are more efficient on constrained hardware), limited data scenarios (CNN inductive biases provide stronger regularization), and real-time latency requirements. For large-scale pretraining, multi-modal tasks, or when pushing absolute accuracy, ViTs win.

### System Design

6. **Design a product image classification system for an e-commerce platform.** Use a pretrained CLIP or DINOv2 ViT as the backbone (rich visual features from web-scale pretraining). Fine-tune on the platform's product taxonomy. For mobile app inference, distill into an EfficientNet-B0. Use caching, batching, and quantization (INT8) for production serving.

7. **How does transfer learning work, and when does it fail?** Early CNN layers learn universal features (edges, textures) that transfer across domains. You freeze these layers and retrain task-specific later layers. It fails when the domain gap is too large (e.g., ImageNet to microscopy) — the early features are not universal for that domain. Solution: find or create domain-specific pretrained models.

---

## Key Takeaways

1. Convolution = sliding pattern detector with parameter sharing across spatial positions
2. CNNs encode locality and translation equivariance — powerful inductive biases for images
3. Feature hierarchy emerges naturally: edges -> textures -> parts -> objects
4. ResNet's skip connections (`output = F(x) + x`) solved training deep networks and appear in every modern architecture
5. Transfer learning from pretrained models is the default starting point for any vision task
6. ViTs have overtaken CNNs at scale, but CNNs remain the practical choice for edge/mobile and limited-data scenarios
7. The line between CNNs and transformers is blurring — hybrid approaches and modernized CNNs (ConvNeXt) compete with pure ViTs
8. Always start with a pretrained model. Training from scratch is almost never the right choice unless you have millions of images

## Summary

CNNs exploit two fundamental properties of images -- locality and translation equivariance -- through parameter-sharing convolution, enabling efficient learning of hierarchical spatial features from edges to objects. The most impactful architectural innovation is the residual connection (output = F(x) + x), which solved the depth problem and appears in every modern architecture. In 2026, start with a pretrained model (CNN or ViT depending on your constraints) and fine-tune -- training from scratch is almost never the right first move.

## What's Next

- **Next lesson:** [RNNs and LSTMs](../rnns-lstms/COURSE.md) -- moves from spatial data to sequential data, exploring how recurrent architectures process variable-length sequences and maintain memory
- **Builds on this:** [Transformers and Attention](../transformers-attention/COURSE.md) -- Vision Transformers (ViT) apply the transformer architecture to images, and understanding CNN inductive biases helps you understand what ViTs trade away for global attention
