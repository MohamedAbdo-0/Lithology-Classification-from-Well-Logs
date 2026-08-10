import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# 1) تحميل البيانات
df = pd.read_csv(r"C:\Users\Albostan\.gemini\antigravity-ide\scratch\adnoc_lithology_ml\data\unified_well_logs_v2.csv", low_memory=False)

# 2) فلترة الصفوف المُصنّفة بس
labeled = df[df['LITHOLOGY_NAME'].notna()].copy()

print(labeled.shape)
print(labeled['LITHOLOGY_NAME'].value_counts())

# 3) تقسيم Train/Test بالبئر
splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_idx, test_idx = next(splitter.split(labeled, groups=labeled['WELL']))
train, test = labeled.iloc[train_idx], labeled.iloc[test_idx]

print("Train rows:", len(train), "| Test rows:", len(test))

# 4) معالجة القيم الناقصة
features = ['GR', 'NPHI', 'RHOB', 'DT', 'PEF']
X_train = train[features].fillna(train[features].median())
y_train = train['LITHOLOGY_NAME']
X_test = test[features].fillna(train[features].median())
y_test = test['LITHOLOGY_NAME']

print("Missing values in X_train after fill:", X_train.isnull().sum().sum())