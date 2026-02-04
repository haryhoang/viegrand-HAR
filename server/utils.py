def normalize_quaternion(q): # Tilt_change parameter
    norm = np.linalg.norm(q)
    if norm == 0:
        return q
    return q / norm


def quaternion_angle(q1, q2):
    q1 = normalize_quaternion(q1)
    q2 = normalize_quaternion(q2)

    dot = np.dot(q1, q2)
    dot = np.clip(dot, -1.0, 1.0)

    angle = 2 * np.arccos(np.abs(dot))
    return angle  # radian


from scipy.stats import kurtosis, skew
def extract_features_window(df_window, fs=50):
    
    svm = np.sqrt(df_window['ax']**2 + df_window['ay']**2 + df_window['az']**2)
    gyro = np.sqrt(df_window['gx']**2 + df_window['gy']**2 + df_window['gz']**2)

    #Jerk magnitude
    jx = df_window['ax'].diff().fillna(0) * fs
    jy = df_window['ay'].diff().fillna(0) * fs
    jz = df_window['az'].diff().fillna(0) * fs
    jerk_mag = np.sqrt(jx**2 + jy**2 + jz**2)

    feats = {}
    feats['impact_max'] = svm.max()
    
    feats['freefall_min'] = svm.min() 
    
    feats['acc_range'] = feats['impact_max'] - feats['freefall_min']

    feats['acc_sma'] = (
        np.sum(np.abs(df_window['ax'])) +
        np.sum(np.abs(df_window['ay'])) +
        np.sum(np.abs(df_window['az']))
    ) / len(df_window)

    #Jerk for vector magnitude
    feats['jerk_max'] = jerk_mag.max()
    feats['jerk_mean'] = jerk_mag.mean()

    #Thống kê
    feats['acc_mean'] = svm.mean()
    feats['acc_std'] = svm.std()
    feats['acc_kurtosis'] = kurtosis(svm)
    feats['acc_skewness'] = skew(svm)
    feats['acc_post_std'] = svm[len(svm)//2 :].std()

    #gyrosope
    feats['gyro_max'] = gyro.max()
    feats['gyro_mean'] = gyro.mean()

    # Orientation
    n = 5 
    q_start = df_window[['s', 'i', 'j', 'k']].iloc[:n].mean().values
    q_end   = df_window[['s', 'i', 'j', 'k']].iloc[-n:].mean().values

    q_start = normalize_quaternion(q_start)
    q_end   = normalize_quaternion(q_end)

    feats['tilt_angle_change'] = quaternion_angle(q_start, q_end)

    
    return feats 


def extract_peak(data, window=25):
    data = data.copy()
    data['svm'] = np.sqrt(data['ax']**2 + data['ay']**2 + data['az']**2)
  
    peak_index = np.argmax(data['svm'].values)
    sample = []

    for shift in range(-5, 6, 10):
        center = peak_index + shift
        start = center - window
        end   = center + window

        if start < 0 or end > len(data):
            continue

        window_event = data.iloc[start:end].reset_index(drop=True)

        if len(window_event) != 2 * window:
            continue


        data_features = extract_features_window(window_event)
        
        sample.append(data_features)

    if len(sample) > 0:
        return pd.DataFrame(sample)
    else:
        return pd.DataFrame()
