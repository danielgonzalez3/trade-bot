
# Replace these with your actual Coinbase Pro API credentials
PUBLIC = 'your_key'
SECRET = 'your_key'

NEIGHBOURS_COUNT = 8
MAX_BARS_BACK = 2000
FEATURE_COUNT = 5

#type Settings
Settings_source = 0.0
Settings_neighborsCount = 0
Settings_maxBarsBack = 0
Settings_featureCount = 0
Settings_colorCompression = 0
Settings_showExits = False
Settings_useDynamicExits = False

#type Label
Label_long = 1
Label_short = -1
Label_neutral = 0

#type FeatureArrays
FeatureArrays_f1 = []
FeatureArrays_f2 = []
FeatureArrays_f3 = []
FeatureArrays_f4 = []
FeatureArrays_f5 = []

#type FeatureSeries
FeatureSeries_f1 = 0.0
FeatureSeries_f2 = 0.0
FeatureSeries_f3 = 0.0
FeatureSeries_f4 = 0.0
FeatureSeries_f5 = 0.0

#type MLModel
MLModel_firstBarIndex = 0
MLModel_trainingLabels = []
MLModel_loopSize = 0
MLModel_lastDistance = 0.0
MLModel_distancesArray = []
MLModel_predictionsArray = []
MLModel_prediction = 0

#type FilterSettings 
FilterSettings_useVolatilityFilter = False
FilterSettings_useRegimeFilter = False
FilterSettings_useAdxFilter = False
FilterSettings_regimeThreshold = 0.0
FilterSettings_adxThreshold = 0

#type Filter
Filter_volatility = False
Filter_regime = False
Filter_adx = False

#Feature Variables: User-Defined Inputs for calculating Feature Series. 
f1_string = 'RSI'
f1_paramA = 0
f1_paramB = 0
f2_string = 'WT'
f2_paramA = 0
f2_paramB = 0
f3_string = 'CCI'
f3_paramA = 0
f3_paramB = 0
f4_string = 'ADX'
f4_paramA = 0
f4_paramB = 0
f5_string = 'RSI'
f5_paramA = 0
f5_paramB = 0
