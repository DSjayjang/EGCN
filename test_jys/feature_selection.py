from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
import numpy as np
import random
import os
os.environ['R_HOME'] = 'C:\Programming\R\R-4.4.2'

# 재현성 난수 고정
SEED = 100

os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

random.seed(SEED)
np.random.seed(SEED)

# data load
df_name = 'hlg'
df = pd.read_csv('C:\Programming\Github\EGCN\data\\' + df_name + '.csv')

smiles_list = df['smiles'].tolist()

# target 정의
target = df.iloc[:,-1]

print(smiles_list[:5])
print(target[:5])

# 분자 특성 추출 class
class MolecularFeatureExtractor:
    def __init__(self):
        self.descriptors = [desc[0] for desc in Descriptors._descList]

    def extract_molecular_features(self, smiles_list):
        features_dict = {desc: [] for desc in self.descriptors}

        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                for descriptor_name in self.descriptors:
                    descriptor_function = getattr(Descriptors, descriptor_name)
                    try:
                        features_dict[descriptor_name].append(descriptor_function(mol))
                    except:
                        features_dict[descriptor_name].append(None)
            else:
                for descriptor_name in self.descriptors:
                    features_dict[descriptor_name].append(None)

        return pd.DataFrame(features_dict)
    
# 분자 특성 추출 및 데이터프레임 정의
extractor = MolecularFeatureExtractor()
df_all_features = extractor.extract_molecular_features(smiles_list)

df_all_features['target'] = target
df_all_features.head()

num_all_features = df_all_features.shape[1] - 1  # logvp 열 제외
print("초기 변수 개수:", num_all_features)

# NA 확인
df_all_features[df_all_features.isna().any(axis = 1)]

# 결측치가 포함된 feature 개수
print('결측치가 포함된 열 개수:', df_all_features.isna().any(axis = 0).sum(), '\n')
print(df_all_features.isna().any(axis = 0))

# 결측치가 포함된 feature 제거
df_removed_features = df_all_features.dropna(axis = 1)
num_removed_features = df_removed_features.shape[1] - 1  # logvp 열 제외

print("제거 후 남은 feature 개수:", num_removed_features)

unique_columns = list(df_removed_features.loc[:, df_removed_features.nunique() == 1].columns)
print('nunique == 1인 feature : \n', unique_columns, '\n')

# nunique == 1인 feature 제거
#df_removed_features.drop(columns = unique_columns, inplace = True)
df_removed_features = df_removed_features.drop(columns = unique_columns).copy()

num_removed_features = df_removed_features.shape[1] - 1  # logvp 열 제외

print("제거 후 남은 feature 개수:", num_removed_features, '\n')
print(df_removed_features.shape)


df_removed_features

# 너무 낮은 variance
low_variances = sorted(df_removed_features.var())
low_variances[:10]

columns_low_variances = []

for i in low_variances:
    if i < 0.001:
        column = df_removed_features.loc[:, df_removed_features.var() == i].columns
        columns_low_variances.append(column)
columns_low_variances = [item for index in columns_low_variances for item in index]

# 2. 중복 제거 및 유니크 값 추출
columns_low_variances = list(set(columns_low_variances))
print(columns_low_variances)

# 낮은 분산의 변수 제거
df_removed_features = df_removed_features.drop(columns = columns_low_variances).copy()
num_removed_features = df_removed_features.shape[1] - 1  # logvp 열 제외

print("제거 후 남은 feature 개수:", num_removed_features, '\n')
print(df_removed_features.shape)



# 데이터 스크리닝
X_train = df_removed_features.drop(columns = 'target')
y_train = df_removed_features['target']

print(X_train.shape)
print(y_train.shape)

# 스케일링
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaling = scaler.transform(X_train)
print(X_train_scaling.shape)

# Python to R type
from rpy2.robjects import r
from rpy2.robjects import pandas2ri
from rpy2.robjects import FloatVector

pandas2ri.activate()

X_train_scaling = r['as.matrix'](X_train_scaling)

y_train = FloatVector(y_train)

nfolds = 10
nfolds = FloatVector([nfolds])[0]

nsis = 100
nsis = FloatVector([nsis])[0]

seed = 9
seed = FloatVector([seed])[0]

from rpy2.robjects.packages import importr
import sys
import io

SIS = importr('SIS')

# R 출력이 발생할 때 UTF-8 오류를 방지하기 위해, 표준 출력을 임시로 바꿔서 처리할 수 있습니다.
#r('Sys.setlocale("LC_ALL", "C.UTF-8")')

# model1 = SIS(...)
model1 = SIS.SIS(X_train_scaling,y_train,
    family="gaussian",
#    penalty="MCP",
    tune="cv",
    nfolds=nfolds,
    nsis=nsis,
    varISIS="aggr",
    seed=seed,
    q = 0.95,
    standardize=False)



# 선택된 feature들의 index
selected_features_ISIS = np.array(model1.rx2('ix'))

# R은 index가 1부터 시작하므로 python에 맞게 보정
selected_features_ISIS = selected_features_ISIS - 1
selected_features_ISIS

df_removed_features_columns = df_removed_features.columns
print(f'ISIS 적용 전 features: {df_removed_features_columns.size}개')
print(df_removed_features_columns, '\n')

selected_features = df_removed_features_columns[selected_features_ISIS]
print(f'ISIS 적용 후 features: {selected_features.size}개')
print(selected_features)

df_ISIS = df_removed_features[list(selected_features) + ['target']]
df_ISIS

# 엘라스틱 넷
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFE

X_ISIS = df_ISIS.drop(columns = ['target'])
y_ISIS = df_ISIS['target']

# train / test split
X_train, X_test, y_train, y_test = train_test_split(X_ISIS, y_ISIS, test_size = 0.2, random_state = SEED)

# scaling
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaling = scaler.transform(X_train)
X_test_scaling = scaler.transform(X_test)

print(X_train_scaling.shape)
print(X_test_scaling.shape)

# ElasticNet 모델과 하이퍼파라미터 범위 설정
elastic_net = ElasticNet(max_iter = 5000)
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],  # 정규화 강도
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]  # L1과 L2 비율
}

kfold = KFold(n_splits = 5, shuffle = True, random_state = SEED)

# GridSearchCV를 사용하여 최적 하이퍼파라미터 탐색
grid_search = GridSearchCV(
    estimator = elastic_net,
    param_grid = param_grid,
    scoring = 'neg_mean_squared_error',
    cv = kfold,
    verbose = 1,
    n_jobs = -1
)

grid_search.fit(X_train_scaling, y_train)

best_params = grid_search.best_params_
print(best_params)

# 최적 하이퍼파라미터로 ElasticNet 모델 생성
best_params = grid_search.best_params_

best_elastic_net = ElasticNet(
    alpha = best_params['alpha'],
    l1_ratio = best_params['l1_ratio'],
    max_iter = 5000,
    fit_intercept=True
)

best_elastic_net.fit(X_train_scaling, y_train)

# 적합
best_elastic_net.fit(X_train_scaling, y_train)

coefficients = best_elastic_net.coef_
coefficients.size

# 엘라스틱넷 적합이후 모든 변수
print(f'# {len(X_train.loc[:, best_elastic_net.coef_ != 0].columns)}개')
print(f'{df_name}_all =', list(X_train.loc[:, best_elastic_net.coef_ != 0].columns), '\n')

from sklearn.inspection import permutation_importance

r = permutation_importance(best_elastic_net, X_test_scaling, y_test,
                           n_repeats=100,
                           random_state=0)
for i in r.importances_mean.argsort()[::-1]:
    if r.importances_mean[i] - 1.96 * r.importances_std[i] > 0:
        print(f"{X_train.columns[i]:<30} \t"
              f"{r.importances_mean[i]:.3f}"
              f" +/- {r.importances_std[i]:.3f}")

from scipy.stats import norm

# 순열 검정 수행
permutation_scores = []
for _ in range(1000):  # 1000회 순열
    shuffled_y = np.random.permutation(y_test)
    score = permutation_importance(best_elastic_net, X_test_scaling, shuffled_y,
                                   n_repeats = 30, random_state = SEED)
    permutation_scores.append(score.importances_mean)

# 귀무가설 하 분포 생성
null_distribution = np.array(permutation_scores)


# p-value 계산
p_values = []
for i, mean in enumerate(r.importances_mean):
    # 단측 검정: 귀무가설 하 중요도 > 관측 중요도
    p_value = (null_distribution[:, i] >= mean).mean()
    p_values.append(p_value)

df_pvalues = pd.DataFrame()
col = []
pval = []
imp = []

for i, p in enumerate(p_values):
    col.append(X_train.columns[i])
    imp.append(r.importances_mean[i])
    pval.append(p)

df_pvalues['Feature'] = col
df_pvalues['Importance'] = imp
df_pvalues['p-value'] = pval

df_pvalues = df_pvalues.sort_values(by = 'Importance', ascending = False)
print(df_pvalues)

# 유의수준 0.05 이하인 변수 - 중요도 순으로 정렬 후 출력
df_pvalues_005 = df_pvalues[df_pvalues['p-value'] <= 0.05]
df_pvalues_005 = df_pvalues_005.sort_values(by = 'Importance', ascending = False)

print(df_pvalues_005)

# 최종 변수 출력
num_features = [3, 5, 7, 10, 20]

for i in num_features:
    print(f'{df_name}_{i} =', list(df_pvalues_005['Feature'][: i]))

print(f'# {len(df_pvalues_005)}개')
print(f'{df_name}_elastic =', list(df_pvalues_005['Feature']))

# # 예측값
# y_pred = best_elastic_net.predict(X_train_scaling)
# coefficients = best_elastic_net.coef_

# # 잔차
# residuals = y_train - y_pred
# # 잔차 제곱합 SSE
# SSE = np.sum(residuals**2)
# # n-p-1
# n_p_1 = len(y_train) - X_train_scaling.shape[1] - 1
# # 잔차의 표준편차 / 오차분산의 불편추정치
# residual_std = np.sqrt(SSE / n_p_1)

# # 표준 오차 계산
# # (X^{T} * X)^{-1}의 대각선 값 추출
# X = np.array(X_train_scaling)
# XtX_inv_diag = np.diag(np.linalg.inv(np.dot(X.T, X)))
# # 표준 오차
# standard_errors = residual_std * np.sqrt(XtX_inv_diag)

# # t-통계량 계산
# t_statistics = best_elastic_net.coef_ / standard_errors

# # elastic 모형
# e_model = pd.DataFrame({'feature' : X_train.columns,
#                         'coef' : coefficients,
#                         't-value' : t_statistics,
#                         'abs(t-value)' : abs(t_statistics)})
# e_model = e_model.sort_values(by='abs(t-value)', ascending = False)
# e_model


# e_model[e_model['abs(t-value)'] > 3].index

# # t-통계량이 3 이상인 변수만 출력
# final_selected_features_index = e_model[e_model['abs(t-value)'] > 3].index

# # 최종 변수 출력
# num_features = [3, 5, 7, 10, 20]

# for i in num_features:
#     print(f'{df_name}_{i} =', list(X_train.columns[final_selected_features_index[: i]]))

# print(f'\n#{len(e_model[e_model["abs(t-value)"] > 3].feature)}개')
# print(f'{df_name}_elastic =', list(e_model[e_model['abs(t-value)'] > 3].feature))
