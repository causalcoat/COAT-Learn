"""Explicit legacy dataset resampling; discovery does not apply this implicitly."""
import pandas as pd
from sklearn.utils import resample

def create_balanced_subset(data, T_name, Y_name, balance_method='both', random_state=42):
    """
    从原始数据中创建一个Y值平衡的子集
    
    Args:
        data: 原始数据DataFrame
        T_name: 处理变量名称
        Y_name: 结果变量名称
        balance_method: 平衡方法，可选'undersample'(欠采样多数类),'oversample'(过采样少数类),'both'(两者结合)
        random_state: 随机种子
        
    Returns:
        平衡后的数据子集
    """
    # 分离多数类和少数类
    majority_class = data[Y_name].value_counts().idxmax()
    minority_class = data[Y_name].value_counts().idxmin()
    
    df_majority = data[data[Y_name] == majority_class]
    df_minority = data[data[Y_name] == minority_class]
    
    print(f"原始数据: 多数类({majority_class})样本数: {len(df_majority)}, 少数类({minority_class})样本数: {len(df_minority)}")
    
    if balance_method == 'undersample':
        # 欠采样多数类
        df_majority_downsampled = resample(
            df_majority, 
            replace=False,
            n_samples=len(df_minority),
            random_state=random_state
        )
        df_balanced = pd.concat([df_majority_downsampled, df_minority])
    
    elif balance_method == 'oversample':
        # 过采样少数类
        df_minority_upsampled = resample(
            df_minority,
            replace=True,
            n_samples=len(df_majority),
            random_state=random_state
        )
        df_balanced = pd.concat([df_majority, df_minority_upsampled])
    
    else:  # 'both'
        # 两者结合: 欠采样多数类并轻微过采样少数类
        target_samples = int((len(df_majority) + len(df_minority)) / 2)
        
        df_majority_downsampled = resample(
            df_majority,
            replace=False,
            n_samples=target_samples,
            random_state=random_state
        )
        
        # 如果少数类样本太少，需要过采样
        if len(df_minority) < target_samples:
            df_minority_upsampled = resample(
                df_minority,
                replace=True,
                n_samples=target_samples,
                random_state=random_state
            )
            df_balanced = pd.concat([df_majority_downsampled, df_minority_upsampled])
        else:
            # 否则也欠采样少数类
            df_minority_downsampled = resample(
                df_minority,
                replace=False,
                n_samples=target_samples,
                random_state=random_state
            )
            df_balanced = pd.concat([df_majority_downsampled, df_minority_downsampled])
    
    # 打乱数据顺序
    df_balanced = df_balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    print(f"平衡后数据: 多数类样本数: {sum(df_balanced[Y_name] == majority_class)}, "
          f"少数类样本数: {sum(df_balanced[Y_name] == minority_class)}")
    print(f"平衡比例: {sum(df_balanced[Y_name] == majority_class) / sum(df_balanced[Y_name] == minority_class):.2f}")

    # 检查T的分布是否也发生了变化
    print("\nT的分布变化:")
    print("原始数据T分布:")
    print(data[T_name].value_counts(normalize=True))
    print("平衡后T分布:")
    print(df_balanced[T_name].value_counts(normalize=True))

    return df_balanced

