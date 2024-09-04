# MobileNet with PyTorch

このリポジトリには、PyTorchを使用して実装されたMobileNetが含まれています。MobileNetは、モバイルおよび組み込みビジョンアプリケーション向けに設計された効率的なディープラーニングアーキテクチャです。深層分離可能な畳み込みを採用することで、パラメータ数と計算コストを削減し、リソースが限られたデバイスに最適です。

## 目次

- [MobileNet with PyTorch](#mobilenet-with-pytorch)
  - [目次](#目次)
  - [イントロダクション](#イントロダクション)
  - [特徴](#特徴)
  - [必要要件](#必要要件)
  - [インストール](#インストール)
  - [使い方](#使い方)
    - [トレーニング](#トレーニング)
    - [評価](#評価)
    - [推論](#推論)
  - [学習済みモデル](#学習済みモデル)
  - [結果](#結果)
  - [参考文献](#参考文献)

## イントロダクション

MobileNetは、Googleによって導入された軽量なディープニューラルネットワークアーキテクチャです。深層分離可能な畳み込みを使用することで、必要なパラメータ数と計算資源を大幅に削減し、モバイルおよび組み込みビジョンアプリケーションに最適です。

## 特徴

- PyTorchで実装されたMobileNet V1アーキテクチャ
- 幅の倍率や解像度の倍率など、カスタマイズ可能なハイパーパラメータ
- スクラッチからのトレーニング、または事前学習済みの重みを使用したファインチューニングが可能
- トレーニング、評価、推論のための使いやすいスクリプト
- CIFAR-10、CIFAR-100、ImageNetデータセットでのサンプルトレーニング

## 必要要件

- Python 3.7以降
- PyTorch 1.7以降
- torchvision
- numpy
- matplotlib
- tqdm

必要なパッケージは以下のコマンドでインストールできます：

```bash
pip install torch torchvision numpy matplotlib tqdm
```

## インストール

リポジトリをローカルマシンにクローンします：

```bash
git clone 
cd 
```

## 使い方

### トレーニング

データセットでMobileNetをトレーニングするには、次のコマンドを使用します：

```bash
python train.py --dataset CIFAR10 --batch-size 64 --epochs 100 --learning-rate 0.01
```

使用可能なオプション:

- `--dataset`: データセットを指定（CIFAR10, CIFAR100, ImageNet など）
- `--batch-size`: トレーニング時のバッチサイズ
- `--epochs`: トレーニングエポック数
- `--learning-rate`: 初期学習率

### 評価

トレーニング済みモデルをテストセットで評価するには：

```bash
python evaluate.py --dataset CIFAR10 --model-path ./models/mobilenet_cifar10.pth
```

使用可能なオプション:

- `--dataset`: 評価用のデータセット（CIFAR10, CIFAR100, ImageNet など）
- `--model-path`: トレーニング済みモデルのパス

### 推論

画像に対して推論を行うには：

```bash
python inference.py --image-path ./path_to_image.jpg --model-path ./models/mobilenet_cifar10.pth
```

## 学習済みモデル

以下の学習済みモデルがダウンロード可能です：

- [MobileNet V1 - CIFAR-10](link_to_pretrained_model)
- [MobileNet V1 - ImageNet](link_to_pretrained_model)

## 結果

実装されたMobileNetの各データセットでのパフォーマンスを以下に示します：

| データセット  | 精度 |
|----------|----------|
| CIFAR-10 | 91.5%    |
| CIFAR-100| 68.3%    |
| ImageNet | 70.6%    |

## 参考文献

- [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
