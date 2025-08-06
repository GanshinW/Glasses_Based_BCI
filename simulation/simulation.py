import numpy as np
import os

def generate_fake_chewing_dataset():
    """
    生成模拟的咀嚼识别数据集用于测试
    """
    # 数据集参数
    n_trials = 300
    n_channels = 4  # F7, F8, T3, T4
    n_samples = 750  # 3秒 × 250Hz
    sampling_rate = 250
    trial_duration = 3.0
    
    # 通道名称
    channel_names = ['F7', 'F8', 'T3', 'T4']
    
    # 生成标签 (平衡数据集)
    labels = np.array([0] * 150 + [1] * 150)  # 150个静息 + 150个咀嚼
    np.random.shuffle(labels)  # 随机打乱
    
    # 生成模拟数据
    trials = []
    
    for i in range(n_trials):
        if labels[i] == 0:  # 静息状态
            # 生成相对平稳的EEG信号
            trial = generate_baseline_signal(n_channels, n_samples)
        else:  # 咀嚼状态  
            # 生成带有咀嚼伪迹的信号
            trial = generate_chewing_signal(n_channels, n_samples)
        
        trials.append(trial)
    
    trials = np.array(trials)  # shape: (300, 4, 750)
    
    # 数据集划分
    indices = np.arange(n_trials)
    np.random.shuffle(indices)
    
    # 70% 训练，15% 验证，15% 测试
    n_train = int(0.7 * n_trials)  # 210
    n_val = int(0.15 * n_trials)   # 45
    n_test = n_trials - n_train - n_val  # 45
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    
    # 准备保存的数据
    dataset = {
        'X_train': trials[train_idx],
        'y_train': labels[train_idx],
        'X_val': trials[val_idx],
        'y_val': labels[val_idx], 
        'X_test': trials[test_idx],
        'y_test': labels[test_idx],
        'channel_names': channel_names,
        'sampling_rate': sampling_rate,
        'trial_duration': trial_duration,
        'n_trials': n_trials,
        'n_channels': n_channels,
        'n_samples': n_samples
    }
    
    return dataset

def generate_baseline_signal(n_channels, n_samples):
    """
    生成静息状态的EEG信号
    """
    # 基础随机噪声
    signal = np.random.randn(n_channels, n_samples) * 20  # 20μV标准差
    
    # 添加一些低频成分 (类似真实EEG的alpha波等)
    t = np.linspace(0, 3, n_samples)
    for ch in range(n_channels):
        # Alpha波 (8-13 Hz)
        alpha_freq = np.random.uniform(8, 13)
        alpha_amp = np.random.uniform(10, 30)
        signal[ch] += alpha_amp * np.sin(2 * np.pi * alpha_freq * t)
        
        # Beta波 (13-30 Hz)  
        beta_freq = np.random.uniform(13, 30)
        beta_amp = np.random.uniform(5, 15)
        signal[ch] += beta_amp * np.sin(2 * np.pi * beta_freq * t)
    
    return signal

def generate_chewing_signal(n_channels, n_samples):
    """
    生成咀嚼状态的EEG信号 (主要在T3/T4通道添加EMG伪迹)
    """
    # 从基础信号开始
    signal = generate_baseline_signal(n_channels, n_samples)
    
    # 在T3/T4通道 (索引2,3) 添加咀嚼EMG伪迹
    t = np.linspace(0, 3, n_samples)
    
    # 咀嚼频率约1-3 Hz (每秒1-3次咀嚼)
    chew_freq = np.random.uniform(1.5, 2.5)
    
    for ch_idx in [2, 3]:  # T3, T4通道
        # 主要咀嚼信号 (低频)
        chew_base = 80 * np.sin(2 * np.pi * chew_freq * t)
        
        # 高频肌电成分 (30-100 Hz)
        emg_freq = np.random.uniform(40, 80)
        emg_signal = 40 * np.sin(2 * np.pi * emg_freq * t)
        
        # 调制高频信号 (使其跟随咀嚼节律)
        modulated_emg = emg_signal * (1 + 0.8 * np.sin(2 * np.pi * chew_freq * t))
        
        # 添加到信号中
        signal[ch_idx] += chew_base + modulated_emg
        
        # 增加这些通道的整体噪声 (肌肉活动)
        signal[ch_idx] += np.random.randn(n_samples) * 15
    
    return signal

def save_dataset(dataset, filename='chewing_dataset.npz'):
    """
    保存数据集到文件
    """
    np.savez_compressed(filename, **dataset)
    print(f"数据集已保存到: {filename}")
    
    # 打印数据集信息
    print(f"\n=== 数据集信息 ===")
    print(f"总试验数: {dataset['n_trials']}")
    print(f"通道数: {dataset['n_channels']} {dataset['channel_names']}")
    print(f"采样点数: {dataset['n_samples']} (时长: {dataset['trial_duration']}秒)")
    print(f"采样率: {dataset['sampling_rate']} Hz")
    print(f"\n=== 数据划分 ===")
    print(f"训练集: {len(dataset['X_train'])}个试验")
    print(f"  - 静息: {np.sum(dataset['y_train'] == 0)}个")
    print(f"  - 咀嚼: {np.sum(dataset['y_train'] == 1)}个")
    print(f"验证集: {len(dataset['X_val'])}个试验") 
    print(f"  - 静息: {np.sum(dataset['y_val'] == 0)}个")
    print(f"  - 咀嚼: {np.sum(dataset['y_val'] == 1)}个")
    print(f"测试集: {len(dataset['X_test'])}个试验")
    print(f"  - 静息: {np.sum(dataset['y_test'] == 0)}个") 
    print(f"  - 咀嚼: {np.sum(dataset['y_test'] == 1)}个")
    print(f"\n=== 数据形状 ===")
    print(f"X_train: {dataset['X_train'].shape}")
    print(f"X_val: {dataset['X_val'].shape}")
    print(f"X_test: {dataset['X_test'].shape}")

def load_and_verify_dataset(filename='chewing_dataset.npz'):
    """
    加载并验证数据集
    """
    data = np.load(filename)
    
    print(f"\n=== 加载验证 ===")
    print("文件中包含的键:")
    for key in data.files:
        if key.startswith(('X_', 'y_')):
            print(f"  {key}: {data[key].shape}")
        else:
            print(f"  {key}: {data[key]}")
    
    # 简单的数据质量检查
    X_train = data['X_train']
    y_train = data['y_train']
    
    print(f"\n=== 数据质量检查 ===")
    print(f"训练数据范围: [{X_train.min():.2f}, {X_train.max():.2f}] μV")
    print(f"标签分布: {np.bincount(y_train)}")
    
    # 检查咀嚼和静息信号的差异
    baseline_trials = X_train[y_train == 0]
    chewing_trials = X_train[y_train == 1]
    
    print(f"\n=== 信号特征对比 ===")
    print("T3通道 (索引2) 平均振幅:")
    print(f"  静息状态: {np.mean(np.abs(baseline_trials[:, 2, :])):.2f} μV") 
    print(f"  咀嚼状态: {np.mean(np.abs(chewing_trials[:, 2, :])):.2f} μV")
    print("T4通道 (索引3) 平均振幅:")
    print(f"  静息状态: {np.mean(np.abs(baseline_trials[:, 3, :])):.2f} μV")
    print(f"  咀嚼状态: {np.mean(np.abs(chewing_trials[:, 3, :])):.2f} μV")

if __name__ == "__main__":
    # 设置随机种子以便复现
    np.random.seed(42)
    
    print("正在生成模拟咀嚼数据集...")
    dataset = generate_fake_chewing_dataset()
    
    # 保存数据集
    save_dataset(dataset)
    
    # 验证加载
    load_and_verify_dataset()
    
    print(f"\n✅ 模拟数据集生成完成！")
    print(f"现在你可以用这个 'chewing_dataset.npz' 文件来测试你的模型了。")