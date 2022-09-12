{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Predicting heart disease using machine learning\n",
    "\n",
    "This notebook looks into using various Python-based machine learning and data science libraries in an attempt to build a machine learning model capable of predicting whether or not someone has heart disease based on their medical attributes.\n",
    "\n",
    "We're going to take the following approach:\n",
    "1. Problem definition\n",
    "2. Data\n",
    "3. Evaluation\n",
    "4. Features\n",
    "5. Modelling\n",
    "6. Experimentation\n",
    "\n",
    "## 1. Problem Definition\n",
    "\n",
    "In a statement,\n",
    "> Given clinical parameters about a patient, can we predict whether or not they have heart disease?\n",
    "\n",
    "## 2. Data\n",
    "\n",
    "The original data came from the Cleavland data from the UCI Machine Learning Repository. https://archive.ics.uci.edu/ml/datasets/heart+Disease\n",
    "\n",
    "There is also a version of it available on Kaggle. https://www.kaggle.com/datasets/sumaiyatasmeem/heart-disease-classification-dataset\n",
    "\n",
    "## 3. Evaluation\n",
    "\n",
    "> If we can reach 95% accuracy at predicting whether or not a patient has heart disease during the proof of concept, we'll pursue the project.\n",
    "\n",
    "## 4. Features\n",
    "\n",
    "This is where you'll get different information about each of the features in your data. You can do this via doing your own research (such as looking at the links above) or by talking to a subject matter expert (someone who knows about the dataset).\n",
    "\n",
    "**Create data dictionary**\n",
    "\n",
    "1. age - age in years\n",
    "2. sex - (1 = male; 0 = female)\n",
    "3. cp - chest pain type\n",
    "    * 0: Typical angina: chest pain related decrease blood supply to the heart\n",
    "    * 1: Atypical angina: chest pain not related to heart\n",
    "    * 2: Non-anginal pain: typically esophageal spasms (non heart related)\n",
    "    * 3: Asymptomatic: chest pain not showing signs of disease\n",
    "4. trestbps - resting blood pressure (in mm Hg on admission to the hospital) anything above 130-140 is typically cause for concern\n",
    "5. chol - serum cholestoral in mg/dl\n",
    "    * serum = LDL + HDL + .2 * triglycerides\n",
    "    * above 200 is cause for concern\n",
    "6. fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)\n",
    "    * '>126' mg/dL signals diabetes\n",
    "7. restecg - resting electrocardiographic results\n",
    "    * 0: Nothing to note\n",
    "    * 1: ST-T Wave abnormality\n",
    "        * can range from mild symptoms to severe problems\n",
    "        * signals non-normal heart beat\n",
    "    * 2: Possible or definite left ventricular hypertrophy\n",
    "        * Enlarged heart's main pumping chamber\n",
    "8. thalach - maximum heart rate achieved\n",
    "9. exang - exercise induced angina (1 = yes; 0 = no)\n",
    "10. oldpeak - ST depression induced by exercise relative to rest looks at stress of heart during excercise unhealthy heart will stress more\n",
    "11. slope - the slope of the peak exercise ST segment\n",
    "    * 0: Upsloping: better heart rate with excercise (uncommon)\n",
    "    * 1: Flatsloping: minimal change (typical healthy heart)\n",
    "    * 2: Downslopins: signs of unhealthy heart\n",
    "12. ca - number of major vessels (0-3) colored by flourosopy\n",
    "    * colored vessel means the doctor can see the blood passing through\n",
    "    * the more blood movement the better (no clots)\n",
    "13. thal - thalium stress result\n",
    "    * 1,3: normal\n",
    "    * 6: fixed defect: used to be defect but ok now\n",
    "    * 7: reversable defect: no proper blood movement when excercising\n",
    "14. target - have disease or not (1=yes, 0=no) (= the predicted attribute)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Preparing the tools\n",
    "\n",
    "We're going to use pandas, Matplotlib and NumPy for data analysis and manipulation."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import all the tools we need\n",
    "\n",
    "# Regular EDA (exploratory data analysis) and plotting libraries\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "# we want our plots to appear inside the notebook\n",
    "%matplotlib inline \n",
    "\n",
    "# Models from Scikit-Learn\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "\n",
    "# Model Evaluations\n",
    "from sklearn.model_selection import train_test_split, cross_val_score\n",
    "from sklearn.model_selection import RandomizedSearchCV, GridSearchCV\n",
    "from sklearn.metrics import confusion_matrix, classification_report\n",
    "from sklearn.metrics import precision_score, recall_score, f1_score\n",
    "from sklearn.metrics import plot_roc_curve"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Load data "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(303, 14)"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.read_csv(\"heart-disease.csv\")\n",
    "df.shape # (rows, columns)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Data Exploration (exploratory data analysis or EDA)\n",
    "\n",
    "The goal here is to find out more about the data and become a subject matter export on the dataset you're working with. \n",
    "\n",
    "1. What question(s) are you trying to solve?\n",
    "2. What kind of data do we have and how do we treat different types?\n",
    "3. What's missing from the data and how do you deal with it?\n",
    "4. Where are the outliers and why should you care about them?\n",
    "5. How can you add, change or remove features to get more out of your data?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>sex</th>\n",
       "      <th>cp</th>\n",
       "      <th>trestbps</th>\n",
       "      <th>chol</th>\n",
       "      <th>fbs</th>\n",
       "      <th>restecg</th>\n",
       "      <th>thalach</th>\n",
       "      <th>exang</th>\n",
       "      <th>oldpeak</th>\n",
       "      <th>slope</th>\n",
       "      <th>ca</th>\n",
       "      <th>thal</th>\n",
       "      <th>target</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>63</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>145</td>\n",
       "      <td>233</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>150</td>\n",
       "      <td>0</td>\n",
       "      <td>2.3</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>37</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>130</td>\n",
       "      <td>250</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>187</td>\n",
       "      <td>0</td>\n",
       "      <td>3.5</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>41</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>130</td>\n",
       "      <td>204</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>172</td>\n",
       "      <td>0</td>\n",
       "      <td>1.4</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>56</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>120</td>\n",
       "      <td>236</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>178</td>\n",
       "      <td>0</td>\n",
       "      <td>0.8</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>57</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>120</td>\n",
       "      <td>354</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>163</td>\n",
       "      <td>1</td>\n",
       "      <td>0.6</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   age  sex  cp  trestbps  chol  fbs  restecg  thalach  exang  oldpeak  slope  \\\n",
       "0   63    1   3       145   233    1        0      150      0      2.3      0   \n",
       "1   37    1   2       130   250    0        1      187      0      3.5      0   \n",
       "2   41    0   1       130   204    0        0      172      0      1.4      2   \n",
       "3   56    1   1       120   236    0        1      178      0      0.8      2   \n",
       "4   57    0   0       120   354    0        1      163      1      0.6      2   \n",
       "\n",
       "   ca  thal  target  \n",
       "0   0     1       1  \n",
       "1   0     2       1  \n",
       "2   0     2       1  \n",
       "3   0     2       1  \n",
       "4   0     2       1  "
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>sex</th>\n",
       "      <th>cp</th>\n",
       "      <th>trestbps</th>\n",
       "      <th>chol</th>\n",
       "      <th>fbs</th>\n",
       "      <th>restecg</th>\n",
       "      <th>thalach</th>\n",
       "      <th>exang</th>\n",
       "      <th>oldpeak</th>\n",
       "      <th>slope</th>\n",
       "      <th>ca</th>\n",
       "      <th>thal</th>\n",
       "      <th>target</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>298</th>\n",
       "      <td>57</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>140</td>\n",
       "      <td>241</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>123</td>\n",
       "      <td>1</td>\n",
       "      <td>0.2</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>299</th>\n",
       "      <td>45</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>110</td>\n",
       "      <td>264</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>132</td>\n",
       "      <td>0</td>\n",
       "      <td>1.2</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>300</th>\n",
       "      <td>68</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>144</td>\n",
       "      <td>193</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>141</td>\n",
       "      <td>0</td>\n",
       "      <td>3.4</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>301</th>\n",
       "      <td>57</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>130</td>\n",
       "      <td>131</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>115</td>\n",
       "      <td>1</td>\n",
       "      <td>1.2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>302</th>\n",
       "      <td>57</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>130</td>\n",
       "      <td>236</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>174</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     age  sex  cp  trestbps  chol  fbs  restecg  thalach  exang  oldpeak  \\\n",
       "298   57    0   0       140   241    0        1      123      1      0.2   \n",
       "299   45    1   3       110   264    0        1      132      0      1.2   \n",
       "300   68    1   0       144   193    1        1      141      0      3.4   \n",
       "301   57    1   0       130   131    0        1      115      1      1.2   \n",
       "302   57    0   1       130   236    0        0      174      0      0.0   \n",
       "\n",
       "     slope  ca  thal  target  \n",
       "298      1   0     3       0  \n",
       "299      1   0     3       0  \n",
       "300      1   2     3       0  \n",
       "301      1   1     3       0  \n",
       "302      1   1     2       0  "
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.tail()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1    165\n",
       "0    138\n",
       "Name: target, dtype: int64"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Let's find out how many of each class there\n",
    "df[\"target\"].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "df[\"target\"].value_counts().plot(kind=\"bar\", color=[\"salmon\", \"lightblue\"]);"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 303 entries, 0 to 302\n",
      "Data columns (total 14 columns):\n",
      "age         303 non-null int64\n",
      "sex         303 non-null int64\n",
      "cp          303 non-null int64\n",
      "trestbps    303 non-null int64\n",
      "chol        303 non-null int64\n",
      "fbs         303 non-null int64\n",
      "restecg     303 non-null int64\n",
      "thalach     303 non-null int64\n",
      "exang       303 non-null int64\n",
      "oldpeak     303 non-null float64\n",
      "slope       303 non-null int64\n",
      "ca          303 non-null int64\n",
      "thal        303 non-null int64\n",
      "target      303 non-null int64\n",
      "dtypes: float64(1), int64(13)\n",
      "memory usage: 33.3 KB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "age         0\n",
       "sex         0\n",
       "cp          0\n",
       "trestbps    0\n",
       "chol        0\n",
       "fbs         0\n",
       "restecg     0\n",
       "thalach     0\n",
       "exang       0\n",
       "oldpeak     0\n",
       "slope       0\n",
       "ca          0\n",
       "thal        0\n",
       "target      0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Are there any missing values?\n",
    "df.isna().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>sex</th>\n",
       "      <th>cp</th>\n",
       "      <th>trestbps</th>\n",
       "      <th>chol</th>\n",
       "      <th>fbs</th>\n",
       "      <th>restecg</th>\n",
       "      <th>thalach</th>\n",
       "      <th>exang</th>\n",
       "      <th>oldpeak</th>\n",
       "      <th>slope</th>\n",
       "      <th>ca</th>\n",
       "      <th>thal</th>\n",
       "      <th>target</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>303.000000</td>\n",
       "      <td>303.000000</td>\n",
       "      <td>303.000000</td>\n",
       "      <td>303.000000</td>\n",
       "      <td>303.000000</td>\n",
       "      <td>303.000000</td>\n",
       "      <td>303.000000</td>\n",
       "      <td>303.000000</td>\n",
       "      <td>303.000000</td>\n",
       "      <td>303.000000</td>\n",
       "      <td>303.000000</td>\n",
       "      <td>303.000000</td>\n",
       "      <td>303.000000</td>\n",
       "      <td>303.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>54.366337</td>\n",
       "      <td>0.683168</td>\n",
       "      <td>0.966997</td>\n",
       "      <td>131.623762</td>\n",
       "      <td>246.264026</td>\n",
       "      <td>0.148515</td>\n",
       "      <td>0.528053</td>\n",
       "      <td>149.646865</td>\n",
       "      <td>0.326733</td>\n",
       "      <td>1.039604</td>\n",
       "      <td>1.399340</td>\n",
       "      <td>0.729373</td>\n",
       "      <td>2.313531</td>\n",
       "      <td>0.544554</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>9.082101</td>\n",
       "      <td>0.466011</td>\n",
       "      <td>1.032052</td>\n",
       "      <td>17.538143</td>\n",
       "      <td>51.830751</td>\n",
       "      <td>0.356198</td>\n",
       "      <td>0.525860</td>\n",
       "      <td>22.905161</td>\n",
       "      <td>0.469794</td>\n",
       "      <td>1.161075</td>\n",
       "      <td>0.616226</td>\n",
       "      <td>1.022606</td>\n",
       "      <td>0.612277</td>\n",
       "      <td>0.498835</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>29.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>94.000000</td>\n",
       "      <td>126.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>71.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>47.500000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>120.000000</td>\n",
       "      <td>211.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>133.500000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>55.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>130.000000</td>\n",
       "      <td>240.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>153.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.800000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>61.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>140.000000</td>\n",
       "      <td>274.500000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>166.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.600000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>77.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>200.000000</td>\n",
       "      <td>564.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>202.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>6.200000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              age         sex          cp    trestbps        chol         fbs  \\\n",
       "count  303.000000  303.000000  303.000000  303.000000  303.000000  303.000000   \n",
       "mean    54.366337    0.683168    0.966997  131.623762  246.264026    0.148515   \n",
       "std      9.082101    0.466011    1.032052   17.538143   51.830751    0.356198   \n",
       "min     29.000000    0.000000    0.000000   94.000000  126.000000    0.000000   \n",
       "25%     47.500000    0.000000    0.000000  120.000000  211.000000    0.000000   \n",
       "50%     55.000000    1.000000    1.000000  130.000000  240.000000    0.000000   \n",
       "75%     61.000000    1.000000    2.000000  140.000000  274.500000    0.000000   \n",
       "max     77.000000    1.000000    3.000000  200.000000  564.000000    1.000000   \n",
       "\n",
       "          restecg     thalach       exang     oldpeak       slope          ca  \\\n",
       "count  303.000000  303.000000  303.000000  303.000000  303.000000  303.000000   \n",
       "mean     0.528053  149.646865    0.326733    1.039604    1.399340    0.729373   \n",
       "std      0.525860   22.905161    0.469794    1.161075    0.616226    1.022606   \n",
       "min      0.000000   71.000000    0.000000    0.000000    0.000000    0.000000   \n",
       "25%      0.000000  133.500000    0.000000    0.000000    1.000000    0.000000   \n",
       "50%      1.000000  153.000000    0.000000    0.800000    1.000000    0.000000   \n",
       "75%      1.000000  166.000000    1.000000    1.600000    2.000000    1.000000   \n",
       "max      2.000000  202.000000    1.000000    6.200000    2.000000    4.000000   \n",
       "\n",
       "             thal      target  \n",
       "count  303.000000  303.000000  \n",
       "mean     2.313531    0.544554  \n",
       "std      0.612277    0.498835  \n",
       "min      0.000000    0.000000  \n",
       "25%      2.000000    0.000000  \n",
       "50%      2.000000    1.000000  \n",
       "75%      3.000000    1.000000  \n",
       "max      3.000000    1.000000  "
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Heart Disease Frequency according to Sex "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1    207\n",
       "0     96\n",
       "Name: sex, dtype: int64"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.sex.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th>sex</th>\n",
       "      <th>0</th>\n",
       "      <th>1</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>target</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>24</td>\n",
       "      <td>114</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>72</td>\n",
       "      <td>93</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "sex      0    1\n",
       "target         \n",
       "0       24  114\n",
       "1       72   93"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Compare target column with sex column\n",
    "pd.crosstab(df.target, df.sex)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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n/40t912GwDXXHMNp512Gtdddx1f+MIXuPnmm7niiis47rjjOP300wF4wQtewPe//32uuuoqjjjiCP7+7//+Ucc94YQTeMtb3sLcuXM577zzOO6440bkfJyylCRJE85wU5YXX3wx1157Leeeey4AS5YsYcGCBWy88cbMnj2b7bbbDoCnPvWpHHDAAQDstttuXHrppUDv4zwOP/xwFi1axK9+9athP7bi3//937nxxhtXLN9///088MADbLbZZut0PgYySZK0XqgqTj/9dA488MDfap8zZw6PecxjVixvsMEGK5Y32GADli1bBsAb3/hG3vrWt3LwwQczZ84cTj311Ecd49e//jWXX345m2yyyYjWbiDTpPG1mxa1LmHc+YOdt2tdgiSNmAMPPJBPfOIT7Lvvvmy00UbcfPPN7LDDDgNvv2TJkhX9zz777GH7HHDAAXzsYx/jxBNPBODqq69m9913X+favYdMkiStF4477jh22WUXnv3sZ7Prrrvyute9bsXo1yBOPfVUDjvsMF74wheyzTbbDNvnox/9KPPmzWPmzJnssssufPKTnxyR2lNVI7KjFmbNmlXz5s1rXYYmCEfIHs0RMklrY/78+TzjGc9oXca4Ntw1SnJlVc0arr8jZJIkSY0ZyCRJkhozkEmSJDVmIJMkSWrMQCZJktSYgUySJKkxA5kkSZpwkvDa1752xfKyZcuYNm0aL3vZy1a53Zw5c1bbpwU/qV+SJK2Tkf6cx0E+I3HTTTfl+uuv56GHHmKTTTbhkksuWaNP5R9vHCGTJEkT0kEHHcQ3v/lNAM455xyOPPLIFeuuuOIKnv/857PHHnvw/Oc/n5tuuulR2z/44IMcc8wxzJ49mz322IMLLrhgzGofykAmSZImpCOOOIIvf/nLPPzww1x77bXstddeK9Y9/elP57LLLuOqq67ive99L+9617setf373vc+9t13X+bOncull17KiSeeyIMPPjiWp7CCU5aSJGlCmjlzJrfffjvnnHMOL33pS39r3ZIlSzjqqKNYsGABSVi6dOmjtr/44ou58MIL+dCHPgTAww8/zB133NHka6EMZJIkacI6+OCDefvb386cOXO45557VrSffPLJvPjFL+b888/n9ttvZ5999nnUtlXFeeedx8477zyGFQ/PKUtJkjRhHXPMMbz73e9mt912+632JUuWrLjJ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\n",
      "text/plain": [
       "<Figure size 720x432 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Create a plot of crosstab\n",
    "pd.crosstab(df.target, df.sex).plot(kind=\"bar\",\n",
    "                                    figsize=(10, 6),\n",
    "                                    color=[\"salmon\", \"lightblue\"])\n",
    "\n",
    "plt.title(\"Heart Disease Frequency for Sex\")\n",
    "plt.xlabel(\"0 = No Diesease, 1 = Disease\")\n",
    "plt.ylabel(\"Amount\")\n",
    "plt.legend([\"Female\", \"Male\"]);\n",
    "plt.xticks(rotation=0);"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Age vs. Max Heart Rate for Heart Disease"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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WNX6G/j617+egYEBnnvuuSHt3/zmN/nOd77D8ccfz7Zt25g0aRK33HILGzdu5L777sM5x+mnn86dd97JiSeeyJVXXsnee+9Nb28vRx55JO9+97t5/PHHeeqpp3jggQcAeOmllwA477zzuOKKKzj44IO59957+djHPsZvf/vbir2m8dKAbAQDt92060Qs6Otj4LabNCATEZG6Vs2/gc65YW3HH388n/3sZ3n/+9/PGWecQUdHB7fccgu33HILCxcuBGDbtm1s3LiRE088kcsuu4wbb7wRgCeffJKNGzdy6KGH8uijj/LJT36SU045hbe97W1s27aNu+66izPPPHNwX6+++mpFX894aUA2kp4Xs7WLiIjUiyr9DXz00Udpbm5mv/32Y/369YPtX/ziFznllFP49a9/zTHHHMOtt96Kc44LL7yQD3/4w0O2cfvtt3Prrbdy9913M3nyZE4++WR27NjB1KlTWb16NTfffDPf+c53uO6667j00kvZa6+9BvPYaolyyEbSPjVbu4iISL2owt/AzZs385GPfIRPfOITw8pEPPLIIxx++OF84QtfYPHixWzYsIGlS5dy5ZVXsm3bNgCeeuopnnvuOXp6epg6dSqTJ09mw4YN3HPPPQBs2bKFgYEB3v3ud/N3f/d3/PGPf2TPPffkoIMO4qc/TaZfnXOsXr26Yq9pIoJFyMzsQOCHwP7AALDcOfdtM9sbuBaYCTwO/Llz7kVL3o1vA+8AtgPnOOf+GKp/Y2lasmzo/DlAaytNS5bF6pKIiEhVhPob2Nvby4IFC+jr66OlpYUPfOADfPaznx32vEsvvZT/+q//orm5mTlz5rBs2TJ222031q9fz7HHHgvAHnvswY9//GPe/va3c8UVVzBv3jwOPfRQjjnmGCAZsJ177rmDZTX+4R/+AYCrr76aj370o1x88cX09fXx3ve+l/nz50/odVWClZu7rciGzQ4ADnDO/dHMpgArgXcB5wAvOOe+YWZfBKY6575gZu8APkkyIDsa+LZz7ujR9rF48WK3YsWKIP2HNKnxtpuSEG37VJqWLFP+mIiI5NL69euZPXu29/P1N3Biyh1vM1vpnCtb2yNYhMw59wzwTPr1VjNbD8wA3gmcnD7tB8DtwBfS9h+6ZIR4j5ntZWYHpNuJovnwRTr5RESkIelvYHVVJYfMzGYCC4F7gemFQVb6/37p02YATxb9WHfaVrqt88xshZmt2Lx5c8hui4iIiFRF8AGZme0B/Az4tHPu5dGeWqZt2Hyqc265c26xc27xvvvuW6luioiIiEQTdEBmZq0kg7GrnXM3pM2b0vyyQp5ZoRpcN3Bg0Y93AE+H7J+IiIhILQg2IEvvmvwesN45962ih34BnJ1+fTbw86L2syxxDNATM39MREREpFpCFoY9HvgAsNbMChXYvgR8A7jOzD4IdAGFcrm/JrnD8mGSshfnBuybiIiISM0IFiFzzv23c86cc/OccwvSf792zj3vnFvinDs4/f+F9PnOOfdx59wbnHOHO+fC1bMQERGRqjMzLrjggsHvv/nNb/LVr37V++evuuoq9t13XxYuXMjBBx/M0qVLueuuuwYf//KXv8ytt95ayS5XjSr1i4iISFXstttu3HDDDWzZsmXc2/iLv/gLVq1axcaNG/niF7/IGWecMbjs0te+9jXe8pa3VKq7VaUBmYiIiAzT1bOdmx7ZxA0PPcNNj2yiq2f7hLfZ0tLCeeedxyWXXDLssSeeeIIlS5Ywb948lixZQldX15jbe9Ob3sR5553H8uXLATjnnHO4/vrrgWQ9zDlz5jBv3jw+97nPAclyTe9+97s58sgjOfLII/n9738PwH333cdxxx3HwoULOe6443jooYcAWLduHUcddRQLFixg3rx5bNy4EYAf//jHg+0f/vCH6e/vn/Cx0YBMRCSDEH+kRGpNV892Vm3qoXdnsuxQ784BVm3qqcj5/vGPf5yrr76anp6eIe2f+MQnOOuss1izZg3vf//7Of/88722d8QRR7Bhw4YhbS+88AI33ngj69atY82aNfzt3/4tAJ/61Kf4zGc+wx/+8Ad+9rOf8aEPfQiAWbNmceedd7Jq1Sq+9rWv8aUvfQmAK664gk996lPcf//9rFixgo6ODtavX8+1117L73//e+6//36am5u5+uqrJ3pYgib1i4jUlcIfqf60QmLhjxRAZ/vkiD0Tqax1W7YOnucF/S5pn+i5vueee3LWWWdx2WWX0dbWNth+9913c8MNSYWsD3zgA3z+85/32l65JSD33HNPJk2axIc+9CFOOeUUTj31VABuvfVWHnzwwcHnvfzyy2zdupWenh7OPvtsNm7ciJnRl67heeyxx/L1r3+d7u5uzjjjDA4++GBuu+02Vq5cyZFHHgkk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E5GGp2kuEYlNAzIRETTNJSJxacpSREREJDINyEREREQi04BMREREJDLlkImINDgtGyUSnwZkIiINrLBsVKGyfWHZKECDMpEq0pSliEgDG23ZKBGpHg3IREQamJaNEqkNmrIUEcmRSud7adkokdqgK05EJCcK+V6FAVQh36urZ/u4t6llo0RqgwZkIiI5ESLfq7N9Mguntw9GxNpamlg4vV0J/SJVpilLEZGcCJXvpWWjROJThExEJCdGyutSvpdI/ukqFhHJCeV7idQvTVmKiOREYVpRVfVF6o8GZCIiOaJ8L5H6pClLERERkcg0IBMRERGJTAMyERERkcg0IBMRERGJTAMyERERkcg0IBMRERGJTAMyERERkcg0IBMRERGJTAMyERERkcg0IBMRERGJLMqAzMw+Y2brzOwBM/uJmU0ys4PM7F4z22hm15rZa2L0TURERKTaqj4gM7MZwPnAYufcYUAz8F7gH4FLnHMHAy8CH6x230RERERiiDVl2QK0mVkLMBl4BngzcH36+A+Ad0Xqm4iIiEhVVX1A5px7Cvgm0EUyEOsBVgIvOed2pk/rBmaU+3kzO8/MVpjZis2bN1ejyyIiIiJBxZiynAq8EzgIeC2wO7CszFNduZ93zi13zi12zi3ed999w3VUREREpEpiTFm+BXjMObfZOdcH3AAcB+yVTmECdABPR+ibiIiISNXFGJB1AceY2WQzM2AJ8CDwX8B70uecDfw8Qt9EREREqi5GDtm9JMn7fwTWpn1YDnwB+KyZPQzsA3yv2n0TERERiaFl7KdUnnPuK8BXSpofBY6K0B0RERGRqFSpX0RERCQyDchEREREItOATERERCQyDchEREREItOATERERCQyDchEREREItOATERERCQyDchEREREItOATERERCQyDchEREREItOATERERCQyDchEREREIouyuLiIiEi96urZzrotW+ndOUBbSxNzp02hs31y7G5JjdOATEREpEK6erazalMP/S75vnfnAKs29QBoUCaj0pSliIhIhazbsnVwMFbQ75J2kdFoQCYiIlIhvTsHMrWLFGhAJiIiUiFtLeX/rI7ULlKgM0RERKRC5k6bQrMNbWu2pF1kNErqFxERqZBC4r7uspSsNCATERGpoM72yRqASWaashQRERGJTAMyERERkcg0IBMRERGJTAMyERERkcg0IBMRERGJTAMyERERkcg0IBMRERGJTAMyERERkcg0IBMRERGJTAMyERERkcg0IBMRERGJTAMyERERkcjMORe7D+NmZpuBJ2L3o8g0YEvsTsio9B7VPr1HtU/vUe3Te1SbXuec27fcA7kekNUaM1vhnFscux8yMr1HtU/vUe3Te1T79B7lj6YsRURERCLTgExEREQkMg3IKmt57A7ImPQe1T69R7VP71Ht03uUM8ohExEREYlMETIRERGRyDQgGyczm2Rm95nZajNbZ2YXpe0Hmdm9ZrbRzK41s9fE7msjM7NmM1tlZv8n/V7vT40xs8fNbK2Z3W9mK9K2vc3sN+n79Bszmxq7n43MzPYys+vNbIOZrTezY/Ue1Q4zOzS9fgr/XjazT+s9yhcNyMbvVeDNzrn5wALg7WZ2DPCPwCXOuYOBF4EPRuyjwKeA9UXf6/2pTW9yzi0ouk3/i8Bt6ft0W/q9xPNt4D+dc7OA+STXlN6jGuGceyi9fhYAi4DtwI3oPcoVDcjGySW2pd+2pv8c8Gbg+rT9B8C7InRPADPrAE4Bvpt+b+j9yYt3krw/oPcpKjPbEzgR+B6Ac+5PzrmX0HtUq5YAjzjnnkDvUa5oQDYB6XTY/cBzwG+AR4CXnHM706d0AzNi9U+4FPg8MJB+vw96f2qRA24xs5Vmdl7aNt059wxA+v9+0Xonrwc2A99Pp/+/a2a7o/eoVr0X+En6td6jHNGAbAKcc/1piLgDOAqYXe5p1e2VAJjZqcBzzrmVxc1lnqr3J77jnXNHAMuAj5vZibE7JEO0AEcA/+qcWwi8gqa+alKaE3s68NPYfZHsNCCrgDR8fztwDLCXmbWkD3UAT8fq1//f3t28WlmFYRi/7vyAykAKiaKPgxODLEIhMCdSEQQRVqfBpiCk/oQmTQsHzQIhaOLAQYQVkUgUQRI0sUElaFCDijzUUUIQnNSgp8FadgYOAomz9tnv9Zu8nxseeNibm3et/a6J2w88neQX4D3aUOVb2J+5U1W/9e1F2ryXh4ELSe4A6NuL4yqcvBVgpapO9+MPaAHNHs2fJ4FvqupCP7ZHG4iB7Dol2ZFke9+/EXicNtH1FLDcb3sJ+HhMhdNWVa9V1V1VtUR7hP9FVb2A/ZkrSW5OcsvVfeAJ4CxwgtYfsE9DVdUqcD7Jrn7qMeB77NE8mrE2XAn2aEPxxbDXKcmDtEmSm2jB9nhVvZ5kJ+2JzK3At8CLVfXnuEqV5ADwalU9ZX/mS+/HR/1wM/BuVR1OchtwHLgH+BV4vqouDSpz8pI8RPtzzFbgJ+AQ/XcPezQXktwEnAd2VtXlfs7v0QZiIJMkSRrMIUtJkqTBDGSSJEmDGcgkSZIGM5BJkiQNZiCTJEkazEAmaZKSPJOkktw3uhZJMpBJmqoZ8BXtxcGSNJSBTNLkJNlGW17rZXogS3JDkreTnEtyMsknSZb7tb1JvuwLoH92dTkaSfq/GMgkTdFB4NOq+hG4lGQP8CywBDwAvALsA0iyBTgCLFfVXuAocHhE0ZIW1+b/vkWSFs6Mttg8tKW0ZsAW4P2q+htYTXKqX98F7AY+TwJtubTf17dcSYvOQCZpUvr6fo8Cu5MULWAVa2tqXvMR4FxV7VunEiVNkEOWkqZmGThWVfdW1VJV3Q38DPwBPNfnkt0OHOj3/wDsSPLvEGaS+0cULmlxGcgkTc2Ma5+GfQjcCawAZ4F3gNPA5ar6ixbi3kxyBvgOeGT9ypU0Bamq0TVI0lxIsq2qrvRhza+B/VW1OrouSYvPOWSStOZkku3AVuANw5ik9eITMkmSpMGcQyZJkjSYgUySJGkwA5kkSdJgBjJJkqTBDGSSJEmDGcgkSZIG+wcISuQjbOY6TQAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 720x432 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Create another figure\n",
    "plt.figure(figsize=(10, 6))\n",
    "\n",
    "# Scatter with postivie examples\n",
    "plt.scatter(df.age[df.target==1],\n",
    "            df.thalach[df.target==1],\n",
    "            c=\"salmon\")\n",
    "\n",
    "# Scatter with negative examples\n",
    "plt.scatter(df.age[df.target==0],\n",
    "            df.thalach[df.target==0],\n",
    "            c=\"lightblue\")\n",
    "\n",
    "# Add some helpful info\n",
    "plt.title(\"Heart Disease in function of Age and Max Heart Rate\")\n",
    "plt.xlabel(\"Age\")\n",
    "plt.ylabel(\"Max Heart Rate\")\n",
    "plt.legend([\"Disease\", \"No Disease\"]);"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Check the distribution of the age column with a histogram\n",
    "df.age.plot.hist();"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Heart Disease Frequency per Chest Pain Type\n",
    "\n",
    "3. cp - chest pain type\n",
    "    * 0: Typical angina: chest pain related decrease blood supply to the heart\n",
    "    * 1: Atypical angina: chest pain not related to heart\n",
    "    * 2: Non-anginal pain: typically esophageal spasms (non heart related)\n",
    "    * 3: Asymptomatic: chest pain not showing signs of disease"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th>target</th>\n",
       "      <th>0</th>\n",
       "      <th>1</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>cp</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>104</td>\n",
       "      <td>39</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>9</td>\n",
       "      <td>41</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>18</td>\n",
       "      <td>69</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>7</td>\n",
       "      <td>16</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "target    0   1\n",
       "cp             \n",
       "0       104  39\n",
       "1         9  41\n",
       "2        18  69\n",
       "3         7  16"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pd.crosstab(df.cp, df.target)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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Pu+LgKepWswaP3V5HvArqm63OcAf6n014xyqgeA3U70nFwOrNiy/HNiKjrsr29XEddbMPu4CPlk/pkXEC8AY4MYmd3EyVbfkH+pr5RaqDzhANXj9Fqqwfzfwb5l5W73sLOA79fV8YrP11n5Kde09DTxC9Z60RGY+QfXBgr/Vr12j6VTXzwyq9/bwusUYqn9rf6YK0bOprokltWxKnbZwcKMkLRci4nSqgeZf6WjdLq5jZ6rWvX51a5OWAxHxU2BaZp7eMG9X4KLM7FeqLskxZJLUSXX36DFUv8QNY8uJiNiYqjV2q9K1SG3ZZSlJnRDVfcpeovp047mFy1GTIuIsqnu+/TAznyxdj9SWXZaSJEmF2UImSZJUmIFMkiSpsOV6UP96662X/fr1K12GJElShyZMmPBcZrb7HcDLdSDr168f48ePL12GJElShyJisV/9ZpelJElSYQYySZKkwgxkkiRJhS3XY8gkSVJz5s2bx/Tp03nttddKl7LC69WrF3379mXllVduehsDmSRJ3cD06dNZY4016NevHxFRupwVVmby/PPPM336dPr379/0dnZZSpLUDbz22musu+66hrEuFhGsu+66nW6JNJBJktRNGMZa4528zgYySZLUEhHBCSecsGj67LPP5vTTT296+8suu4zevXszZMgQBgwYwO67785dd921aPl3v/tdbrnllqVZcss4hkySpG5o3hkndLxSJ6x82r90uM4qq6zC9ddfz6mnnsp66633jo7zxS9+kQsuuACAcePGMWLECMaNG8fmm2/O9773vXe0z2WBLWSSJKklevbsyahRo/jJT37ytmV//vOf2WWXXRg0aBC77LILTz75ZIf7+8QnPsGoUaMYM2YMAIcccgjXXnstAKeccgoDBw5k0KBBnHjiiQDMmjWL/fbbj2HDhjFs2DDuvPNOAO6991522GEHhgwZwg477MCjjz4KwKRJk9h2220ZPHgwgwYNYurUqQBceeWVi+YfccQRzJ8//12/NgYySZLUMt/85jcZO3Yss2fPfsv8I488koMPPpiHHnqIL3/5yxx99NFN7W+bbbZhypQpb5n3wgsvcMMNNzBp0iQeeughvvOd7wBwzDHHcNxxx3Hfffdx3XXXcfjhhwOw2Wabcccdd/DAAw/wve99j29/+9sAjB49mmOOOYaJEycyfvx4+vbty+TJk7n66qu58847mThxIj169GDs2LHv9mWxy1KSJLXOmmuuycEHH8z555/Pqquuumj+3XffzfXXXw/AQQcdxEknndTU/jKz3WP06tWLww8/nL322ou9994bgFtuuYVHHnlk0Xovv/wyc+bMYfbs2YwcOZKpU6cSEcybNw+A7bffnh/84AdMnz6dESNGMGDAAG699VYmTJjAsGHDAHj11VdZf/3139mL0cBAJkmSWurYY49lm2224dBDD13sOs1+UvGBBx5g8803f8u8nj17cu+993Lrrbdy1VVXccEFF/C73/2OBQsWcPfdd78lCAIcddRRfOITn+CGG27giSeeYPjw4QAceOCBfPSjH+U3v/kNu+++OxdddBGZyciRIznrrLM6d9IdMJAtJUt7cOS70czASkmSSllnnXX4whe+wMUXX8xXv/pVAHbYYQeuuuoqDjroIMaOHctOO+3U4X5uv/12xowZw7hx494yf+7cubzyyit8+tOfZrvttmOTTTYBYLfdduOCCy7gW9/6FgATJ05k8ODBzJ49mz59+gDVJzkXmjZtGhtvvDFHH30006ZN46GHHmK33XZjn3324bjjjmP99dfnhRdeYM6cOWy00Ubv6jVxDJkkSWq5E044geeee27R9Pnnn8+ll17KoEGDuOKKKzjvvPPa3e7qq69m8ODBfPjDH+aHP/wh11133dtayObMmcPee+/NoEGD+PjHP77oQwTnn38+48ePZ9CgQQwcOJDRo0cDcNJJJ3Hqqaey4447vmWA/tVXX82WW27J4MGDmTJlCgcffDADBw7kzDPPZLfddmPQoEF86lOfYsaMGe/69Yj2+l6XF0OHDs3x48eXLgOwhUyStGybPHny24KLuk57r3dETMjMoe2tbwuZJElSYQYySZKkwgxkkiRJhRnIJEmSCjOQSZIkFWYgkyRJKqzLAllEXBIRMyPi4YZ560TEbyNiav1z7Xp+RMT5EfFYRDwUEdt0VV2SJKmMHj16MHjwYLbYYgu23nprzjnnHBYsWADA+PHjm/7+yhVRV96p/zLgAuCnDfNOAW7NzH+KiFPq6ZOBPYEB9eOjwL/XPyVJUhe4/tF3fzPTRiM23bDDdVZddVUmTpwIwMyZMznwwAOZPXs2Z5xxBkOHDmXo0HZv0dUtdFkLWWbeAbzQZvY+wOX188uBfRvm/zQrfwDWioiO31lJkrRcWn/99RkzZgwXXHABmcltt9226EvAb7/9dgYPHszgwYMZMmQIc+bMAeDHP/4xw4YNY9CgQZx22mmL9rXvvvvykY98hC222IIxY8YAMH/+fA455BC23HJLttpqq0V363/88cfZY489+MhHPsLHPvYxpkyZ0uIzb1+rv8tyg8ycAZCZMyJi4dej9wGealhvej3vbfE9IkYBowA++MEPdm21kiSpy2y88cYsWLCAmTNnvmX+2WefzYUXXsiOO+7I3Llz6dWrFzfffDNTp07l3nvvJTP57Gc/yx133MHOO+/MJZdcwjrrrMOrr77KsGHD2G+//XjiiSd4+umnefjhauTUSy+9BMCoUaMYPXo0AwYM4J577uEb3/gGv/vd71p+7m0tK18u3t5Xurf7nU6ZOQYYA9VXJ3VlUZIkqWu19xWOO+64I8cffzxf/vKXGTFiBH379uXmm2/m5ptvZsiQIUD1BeJTp05l55135vzzz+eGG24A4KmnnmLq1KlsuummTJs2jaOOOoq99tqL3Xbbjblz53LXXXex//77LzrW66+/3poT7UCrA9mzEbFh3Tq2IbAwEk8HPtCwXl/gmRbXJkmSWmjatGn06NGD9ddfn8mTJy+af8opp7DXXntx4403st1223HLLbeQmZx66qkcccQRb9nHbbfdxi233MLdd9/NaqutxvDhw3nttddYe+21efDBB7npppu48MILueaaazj33HNZa621Fo1jW5a0+rYXvwJG1s9HAr9smH9w/WnL7YDZC7s2JUnSimfWrFl87Wtf48gjjyTirR1ljz/+OFtttRUnn3wyQ4cOZcqUKey+++5ccsklzJ07F4Cnn36amTNnMnv2bNZee21WW201pkyZwh/+8AcAnnvuORYsWMB+++3H97//fe6//37WXHNN+vfvz89//nOgap178MEHW3vii9FlLWQR8TNgOLBeREwHTgP+CbgmIg4DngQWthneCHwaeAx4BTi0q+qSJEllvPrqqwwePJh58+bRs2dPDjroII4//vi3rXfuuecybtw4evTowcCBA9lzzz1ZZZVVmDx5Mttvvz0Aq6++OldeeSV77LEHo0ePZtCgQWy66aZst912QBXYDj300EW31TjrrLMAGDt2LF//+tc588wzmTdvHgcccABbb711i16BxYv2+m6XF0OHDs3x48eXLgOAeWecULqERVY+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\n",
      "text/plain": [
       "<Figure size 720x432 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Make the crosstab more visual\n",
    "pd.crosstab(df.cp, df.target).plot(kind=\"bar\",\n",
    "                                   figsize=(10, 6),\n",
    "                                   color=[\"salmon\", \"lightblue\"])\n",
    "\n",
    "# Add some communication\n",
    "plt.title(\"Heart Disease Frequency Per Chest Pain Type\")\n",
    "plt.xlabel(\"Chest Pain Type\")\n",
    "plt.ylabel(\"Amount\")\n",
    "plt.legend([\"No Disease\", \"Disease\"])\n",
    "plt.xticks(rotation=0);"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>sex</th>\n",
       "      <th>cp</th>\n",
       "      <th>trestbps</th>\n",
       "      <th>chol</th>\n",
       "      <th>fbs</th>\n",
       "      <th>restecg</th>\n",
       "      <th>thalach</th>\n",
       "      <th>exang</th>\n",
       "      <th>oldpeak</th>\n",
       "      <th>slope</th>\n",
       "      <th>ca</th>\n",
       "      <th>thal</th>\n",
       "      <th>target</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>63</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>145</td>\n",
       "      <td>233</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>150</td>\n",
       "      <td>0</td>\n",
       "      <td>2.3</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>37</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>130</td>\n",
       "      <td>250</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>187</td>\n",
       "      <td>0</td>\n",
       "      <td>3.5</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>41</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>130</td>\n",
       "      <td>204</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>172</td>\n",
       "      <td>0</td>\n",
       "      <td>1.4</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>56</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>120</td>\n",
       "      <td>236</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>178</td>\n",
       "      <td>0</td>\n",
       "      <td>0.8</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>57</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>120</td>\n",
       "      <td>354</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>163</td>\n",
       "      <td>1</td>\n",
       "      <td>0.6</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   age  sex  cp  trestbps  chol  fbs  restecg  thalach  exang  oldpeak  slope  \\\n",
       "0   63    1   3       145   233    1        0      150      0      2.3      0   \n",
       "1   37    1   2       130   250    0        1      187      0      3.5      0   \n",
       "2   41    0   1       130   204    0        0      172      0      1.4      2   \n",
       "3   56    1   1       120   236    0        1      178      0      0.8      2   \n",
       "4   57    0   0       120   354    0        1      163      1      0.6      2   \n",
       "\n",
       "   ca  thal  target  \n",
       "0   0     1       1  \n",
       "1   0     2       1  \n",
       "2   0     2       1  \n",
       "3   0     2       1  \n",
       "4   0     2       1  "
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>sex</th>\n",
       "      <th>cp</th>\n",
       "      <th>trestbps</th>\n",
       "      <th>chol</th>\n",
       "      <th>fbs</th>\n",
       "      <th>restecg</th>\n",
       "      <th>thalach</th>\n",
       "      <th>exang</th>\n",
       "      <th>oldpeak</th>\n",
       "      <th>slope</th>\n",
       "      <th>ca</th>\n",
       "      <th>thal</th>\n",
       "      <th>target</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>age</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.098447</td>\n",
       "      <td>-0.068653</td>\n",
       "      <td>0.279351</td>\n",
       "      <td>0.213678</td>\n",
       "      <td>0.121308</td>\n",
       "      <td>-0.116211</td>\n",
       "      <td>-0.398522</td>\n",
       "      <td>0.096801</td>\n",
       "      <td>0.210013</td>\n",
       "      <td>-0.168814</td>\n",
       "      <td>0.276326</td>\n",
       "      <td>0.068001</td>\n",
       "      <td>-0.225439</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>sex</th>\n",
       "      <td>-0.098447</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.049353</td>\n",
       "      <td>-0.056769</td>\n",
       "      <td>-0.197912</td>\n",
       "      <td>0.045032</td>\n",
       "      <td>-0.058196</td>\n",
       "      <td>-0.044020</td>\n",
       "      <td>0.141664</td>\n",
       "      <td>0.096093</td>\n",
       "      <td>-0.030711</td>\n",
       "      <td>0.118261</td>\n",
       "      <td>0.210041</td>\n",
       "      <td>-0.280937</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>cp</th>\n",
       "      <td>-0.068653</td>\n",
       "      <td>-0.049353</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.047608</td>\n",
       "      <td>-0.076904</td>\n",
       "      <td>0.094444</td>\n",
       "      <td>0.044421</td>\n",
       "      <td>0.295762</td>\n",
       "      <td>-0.394280</td>\n",
       "      <td>-0.149230</td>\n",
       "      <td>0.119717</td>\n",
       "      <td>-0.181053</td>\n",
       "      <td>-0.161736</td>\n",
       "      <td>0.433798</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>trestbps</th>\n",
       "      <td>0.279351</td>\n",
       "      <td>-0.056769</td>\n",
       "      <td>0.047608</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.123174</td>\n",
       "      <td>0.177531</td>\n",
       "      <td>-0.114103</td>\n",
       "      <td>-0.046698</td>\n",
       "      <td>0.067616</td>\n",
       "      <td>0.193216</td>\n",
       "      <td>-0.121475</td>\n",
       "      <td>0.101389</td>\n",
       "      <td>0.062210</td>\n",
       "      <td>-0.144931</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>chol</th>\n",
       "      <td>0.213678</td>\n",
       "      <td>-0.197912</td>\n",
       "      <td>-0.076904</td>\n",
       "      <td>0.123174</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.013294</td>\n",
       "      <td>-0.151040</td>\n",
       "      <td>-0.009940</td>\n",
       "      <td>0.067023</td>\n",
       "      <td>0.053952</td>\n",
       "      <td>-0.004038</td>\n",
       "      <td>0.070511</td>\n",
       "      <td>0.098803</td>\n",
       "      <td>-0.085239</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>fbs</th>\n",
       "      <td>0.121308</td>\n",
       "      <td>0.045032</td>\n",
       "      <td>0.094444</td>\n",
       "      <td>0.177531</td>\n",
       "      <td>0.013294</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.084189</td>\n",
       "      <td>-0.008567</td>\n",
       "      <td>0.025665</td>\n",
       "      <td>0.005747</td>\n",
       "      <td>-0.059894</td>\n",
       "      <td>0.137979</td>\n",
       "      <td>-0.032019</td>\n",
       "      <td>-0.028046</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>restecg</th>\n",
       "      <td>-0.116211</td>\n",
       "      <td>-0.058196</td>\n",
       "      <td>0.044421</td>\n",
       "      <td>-0.114103</td>\n",
       "      <td>-0.151040</td>\n",
       "      <td>-0.084189</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.044123</td>\n",
       "      <td>-0.070733</td>\n",
       "      <td>-0.058770</td>\n",
       "      <td>0.093045</td>\n",
       "      <td>-0.072042</td>\n",
       "      <td>-0.011981</td>\n",
       "      <td>0.137230</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>thalach</th>\n",
       "      <td>-0.398522</td>\n",
       "      <td>-0.044020</td>\n",
       "      <td>0.295762</td>\n",
       "      <td>-0.046698</td>\n",
       "      <td>-0.009940</td>\n",
       "      <td>-0.008567</td>\n",
       "      <td>0.044123</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.378812</td>\n",
       "      <td>-0.344187</td>\n",
       "      <td>0.386784</td>\n",
       "      <td>-0.213177</td>\n",
       "      <td>-0.096439</td>\n",
       "      <td>0.421741</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>exang</th>\n",
       "      <td>0.096801</td>\n",
       "      <td>0.141664</td>\n",
       "      <td>-0.394280</td>\n",
       "      <td>0.067616</td>\n",
       "      <td>0.067023</td>\n",
       "      <td>0.025665</td>\n",
       "      <td>-0.070733</td>\n",
       "      <td>-0.378812</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.288223</td>\n",
       "      <td>-0.257748</td>\n",
       "      <td>0.115739</td>\n",
       "      <td>0.206754</td>\n",
       "      <td>-0.436757</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>oldpeak</th>\n",
       "      <td>0.210013</td>\n",
       "      <td>0.096093</td>\n",
       "      <td>-0.149230</td>\n",
       "      <td>0.193216</td>\n",
       "      <td>0.053952</td>\n",
       "      <td>0.005747</td>\n",
       "      <td>-0.058770</td>\n",
       "      <td>-0.344187</td>\n",
       "      <td>0.288223</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.577537</td>\n",
       "      <td>0.222682</td>\n",
       "      <td>0.210244</td>\n",
       "      <td>-0.430696</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>slope</th>\n",
       "      <td>-0.168814</td>\n",
       "      <td>-0.030711</td>\n",
       "      <td>0.119717</td>\n",
       "      <td>-0.121475</td>\n",
       "      <td>-0.004038</td>\n",
       "      <td>-0.059894</td>\n",
       "      <td>0.093045</td>\n",
       "      <td>0.386784</td>\n",
       "      <td>-0.257748</td>\n",
       "      <td>-0.577537</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.080155</td>\n",
       "      <td>-0.104764</td>\n",
       "      <td>0.345877</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ca</th>\n",
       "      <td>0.276326</td>\n",
       "      <td>0.118261</td>\n",
       "      <td>-0.181053</td>\n",
       "      <td>0.101389</td>\n",
       "      <td>0.070511</td>\n",
       "      <td>0.137979</td>\n",
       "      <td>-0.072042</td>\n",
       "      <td>-0.213177</td>\n",
       "      <td>0.115739</td>\n",
       "      <td>0.222682</td>\n",
       "      <td>-0.080155</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.151832</td>\n",
       "      <td>-0.391724</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>thal</th>\n",
       "      <td>0.068001</td>\n",
       "      <td>0.210041</td>\n",
       "      <td>-0.161736</td>\n",
       "      <td>0.062210</td>\n",
       "      <td>0.098803</td>\n",
       "      <td>-0.032019</td>\n",
       "      <td>-0.011981</td>\n",
       "      <td>-0.096439</td>\n",
       "      <td>0.206754</td>\n",
       "      <td>0.210244</td>\n",
       "      <td>-0.104764</td>\n",
       "      <td>0.151832</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.344029</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>target</th>\n",
       "      <td>-0.225439</td>\n",
       "      <td>-0.280937</td>\n",
       "      <td>0.433798</td>\n",
       "      <td>-0.144931</td>\n",
       "      <td>-0.085239</td>\n",
       "      <td>-0.028046</td>\n",
       "      <td>0.137230</td>\n",
       "      <td>0.421741</td>\n",
       "      <td>-0.436757</td>\n",
       "      <td>-0.430696</td>\n",
       "      <td>0.345877</td>\n",
       "      <td>-0.391724</td>\n",
       "      <td>-0.344029</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "               age       sex        cp  trestbps      chol       fbs  \\\n",
       "age       1.000000 -0.098447 -0.068653  0.279351  0.213678  0.121308   \n",
       "sex      -0.098447  1.000000 -0.049353 -0.056769 -0.197912  0.045032   \n",
       "cp       -0.068653 -0.049353  1.000000  0.047608 -0.076904  0.094444   \n",
       "trestbps  0.279351 -0.056769  0.047608  1.000000  0.123174  0.177531   \n",
       "chol      0.213678 -0.197912 -0.076904  0.123174  1.000000  0.013294   \n",
       "fbs       0.121308  0.045032  0.094444  0.177531  0.013294  1.000000   \n",
       "restecg  -0.116211 -0.058196  0.044421 -0.114103 -0.151040 -0.084189   \n",
       "thalach  -0.398522 -0.044020  0.295762 -0.046698 -0.009940 -0.008567   \n",
       "exang     0.096801  0.141664 -0.394280  0.067616  0.067023  0.025665   \n",
       "oldpeak   0.210013  0.096093 -0.149230  0.193216  0.053952  0.005747   \n",
       "slope    -0.168814 -0.030711  0.119717 -0.121475 -0.004038 -0.059894   \n",
       "ca        0.276326  0.118261 -0.181053  0.101389  0.070511  0.137979   \n",
       "thal      0.068001  0.210041 -0.161736  0.062210  0.098803 -0.032019   \n",
       "target   -0.225439 -0.280937  0.433798 -0.144931 -0.085239 -0.028046   \n",
       "\n",
       "           restecg   thalach     exang   oldpeak     slope        ca  \\\n",
       "age      -0.116211 -0.398522  0.096801  0.210013 -0.168814  0.276326   \n",
       "sex      -0.058196 -0.044020  0.141664  0.096093 -0.030711  0.118261   \n",
       "cp        0.044421  0.295762 -0.394280 -0.149230  0.119717 -0.181053   \n",
       "trestbps -0.114103 -0.046698  0.067616  0.193216 -0.121475  0.101389   \n",
       "chol     -0.151040 -0.009940  0.067023  0.053952 -0.004038  0.070511   \n",
       "fbs      -0.084189 -0.008567  0.025665  0.005747 -0.059894  0.137979   \n",
       "restecg   1.000000  0.044123 -0.070733 -0.058770  0.093045 -0.072042   \n",
       "thalach   0.044123  1.000000 -0.378812 -0.344187  0.386784 -0.213177   \n",
       "exang    -0.070733 -0.378812  1.000000  0.288223 -0.257748  0.115739   \n",
       "oldpeak  -0.058770 -0.344187  0.288223  1.000000 -0.577537  0.222682   \n",
       "slope     0.093045  0.386784 -0.257748 -0.577537  1.000000 -0.080155   \n",
       "ca       -0.072042 -0.213177  0.115739  0.222682 -0.080155  1.000000   \n",
       "thal     -0.011981 -0.096439  0.206754  0.210244 -0.104764  0.151832   \n",
       "target    0.137230  0.421741 -0.436757 -0.430696  0.345877 -0.391724   \n",
       "\n",
       "              thal    target  \n",
       "age       0.068001 -0.225439  \n",
       "sex       0.210041 -0.280937  \n",
       "cp       -0.161736  0.433798  \n",
       "trestbps  0.062210 -0.144931  \n",
       "chol      0.098803 -0.085239  \n",
       "fbs      -0.032019 -0.028046  \n",
       "restecg  -0.011981  0.137230  \n",
       "thalach  -0.096439  0.421741  \n",
       "exang     0.206754 -0.436757  \n",
       "oldpeak   0.210244 -0.430696  \n",
       "slope    -0.104764  0.345877  \n",
       "ca        0.151832 -0.391724  \n",
       "thal      1.000000 -0.344029  \n",
       "target   -0.344029  1.000000  "
      ]
     },
     "execution_count": 46,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Make a correlation matrix\n",
    "df.corr()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(14.0, 0.0)"
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1080x720 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Let's make our correlation matrix a little prettier\n",
    "corr_matrix = df.corr()\n",
    "fig, ax = plt.subplots(figsize=(15, 10))\n",
    "ax = sns.heatmap(corr_matrix,\n",
    "                 annot=True,\n",
    "                 linewidths=0.5,\n",
    "                 fmt=\".2f\",\n",
    "                 cmap=\"YlGnBu\");\n",
    "bottom, top = ax.get_ylim()\n",
    "ax.set_ylim(bottom + 0.5, top - 0.5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Modelling "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>sex</th>\n",
       "      <th>cp</th>\n",
       "      <th>trestbps</th>\n",
       "      <th>chol</th>\n",
       "      <th>fbs</th>\n",
       "      <th>restecg</th>\n",
       "      <th>thalach</th>\n",
       "      <th>exang</th>\n",
       "      <th>oldpeak</th>\n",
       "      <th>slope</th>\n",
       "      <th>ca</th>\n",
       "      <th>thal</th>\n",
       "      <th>target</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>63</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>145</td>\n",
       "      <td>233</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>150</td>\n",
       "      <td>0</td>\n",
       "      <td>2.3</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>37</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>130</td>\n",
       "      <td>250</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>187</td>\n",
       "      <td>0</td>\n",
       "      <td>3.5</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>41</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>130</td>\n",
       "      <td>204</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>172</td>\n",
       "      <td>0</td>\n",
       "      <td>1.4</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>56</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>120</td>\n",
       "      <td>236</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>178</td>\n",
       "      <td>0</td>\n",
       "      <td>0.8</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>57</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>120</td>\n",
       "      <td>354</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>163</td>\n",
       "      <td>1</td>\n",
       "      <td>0.6</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   age  sex  cp  trestbps  chol  fbs  restecg  thalach  exang  oldpeak  slope  \\\n",
       "0   63    1   3       145   233    1        0      150      0      2.3      0   \n",
       "1   37    1   2       130   250    0        1      187      0      3.5      0   \n",
       "2   41    0   1       130   204    0        0      172      0      1.4      2   \n",
       "3   56    1   1       120   236    0        1      178      0      0.8      2   \n",
       "4   57    0   0       120   354    0        1      163      1      0.6      2   \n",
       "\n",
       "   ca  thal  target  \n",
       "0   0     1       1  \n",
       "1   0     2       1  \n",
       "2   0     2       1  \n",
       "3   0     2       1  \n",
       "4   0     2       1  "
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Split data into X and y\n",
    "X = df.drop(\"target\", axis=1)\n",
    "\n",
    "y = df[\"target\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>sex</th>\n",
       "      <th>cp</th>\n",
       "      <th>trestbps</th>\n",
       "      <th>chol</th>\n",
       "      <th>fbs</th>\n",
       "      <th>restecg</th>\n",
       "      <th>thalach</th>\n",
       "      <th>exang</th>\n",
       "      <th>oldpeak</th>\n",
       "      <th>slope</th>\n",
       "      <th>ca</th>\n",
       "      <th>thal</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>63</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>145</td>\n",
       "      <td>233</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>150</td>\n",
       "      <td>0</td>\n",
       "      <td>2.3</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>37</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>130</td>\n",
       "      <td>250</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>187</td>\n",
       "      <td>0</td>\n",
       "      <td>3.5</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>41</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>130</td>\n",
       "      <td>204</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>172</td>\n",
       "      <td>0</td>\n",
       "      <td>1.4</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>56</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>120</td>\n",
       "      <td>236</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>178</td>\n",
       "      <td>0</td>\n",
       "      <td>0.8</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>57</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>120</td>\n",
       "      <td>354</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>163</td>\n",
       "      <td>1</td>\n",
       "      <td>0.6</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>298</th>\n",
       "      <td>57</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>140</td>\n",
       "      <td>241</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>123</td>\n",
       "      <td>1</td>\n",
       "      <td>0.2</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>299</th>\n",
       "      <td>45</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>110</td>\n",
       "      <td>264</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>132</td>\n",
       "      <td>0</td>\n",
       "      <td>1.2</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>300</th>\n",
       "      <td>68</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>144</td>\n",
       "      <td>193</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>141</td>\n",
       "      <td>0</td>\n",
       "      <td>3.4</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>301</th>\n",
       "      <td>57</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>130</td>\n",
       "      <td>131</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>115</td>\n",
       "      <td>1</td>\n",
       "      <td>1.2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>302</th>\n",
       "      <td>57</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>130</td>\n",
       "      <td>236</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>174</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>303 rows × 13 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     age  sex  cp  trestbps  chol  fbs  restecg  thalach  exang  oldpeak  \\\n",
       "0     63    1   3       145   233    1        0      150      0      2.3   \n",
       "1     37    1   2       130   250    0        1      187      0      3.5   \n",
       "2     41    0   1       130   204    0        0      172      0      1.4   \n",
       "3     56    1   1       120   236    0        1      178      0      0.8   \n",
       "4     57    0   0       120   354    0        1      163      1      0.6   \n",
       "..   ...  ...  ..       ...   ...  ...      ...      ...    ...      ...   \n",
       "298   57    0   0       140   241    0        1      123      1      0.2   \n",
       "299   45    1   3       110   264    0        1      132      0      1.2   \n",
       "300   68    1   0       144   193    1        1      141      0      3.4   \n",
       "301   57    1   0       130   131    0        1      115      1      1.2   \n",
       "302   57    0   1       130   236    0        0      174      0      0.0   \n",
       "\n",
       "     slope  ca  thal  \n",
       "0        0   0     1  \n",
       "1        0   0     2  \n",
       "2        2   0     2  \n",
       "3        2   0     2  \n",
       "4        2   0     2  \n",
       "..     ...  ..   ...  \n",
       "298      1   0     3  \n",
       "299      1   0     3  \n",
       "300      1   2     3  \n",
       "301      1   1     3  \n",
       "302      1   1     2  \n",
       "\n",
       "[303 rows x 13 columns]"
      ]
     },
     "execution_count": 52,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0      1\n",
       "1      1\n",
       "2      1\n",
       "3      1\n",
       "4      1\n",
       "      ..\n",
       "298    0\n",
       "299    0\n",
       "300    0\n",
       "301    0\n",
       "302    0\n",
       "Name: target, Length: 303, dtype: int64"
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Split data into train and test sets\n",
    "np.random.seed(42)\n",
    "\n",
    "# Split into train & test set\n",
    "X_train, X_test, y_train, y_test = train_test_split(X,\n",
    "                                                    y,\n",
    "                                                    test_size=0.2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>sex</th>\n",
       "      <th>cp</th>\n",
       "      <th>trestbps</th>\n",
       "      <th>chol</th>\n",
       "      <th>fbs</th>\n",
       "      <th>restecg</th>\n",
       "      <th>thalach</th>\n",
       "      <th>exang</th>\n",
       "      <th>oldpeak</th>\n",
       "      <th>slope</th>\n",
       "      <th>ca</th>\n",
       "      <th>thal</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>132</th>\n",
       "      <td>42</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>120</td>\n",
       "      <td>295</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>162</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>202</th>\n",
       "      <td>58</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>150</td>\n",
       "      <td>270</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>111</td>\n",
       "      <td>1</td>\n",
       "      <td>0.8</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>196</th>\n",
       "      <td>46</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>150</td>\n",
       "      <td>231</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>147</td>\n",
       "      <td>0</td>\n",
       "      <td>3.6</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75</th>\n",
       "      <td>55</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>135</td>\n",
       "      <td>250</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>161</td>\n",
       "      <td>0</td>\n",
       "      <td>1.4</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>176</th>\n",
       "      <td>60</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>117</td>\n",
       "      <td>230</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>160</td>\n",
       "      <td>1</td>\n",
       "      <td>1.4</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>188</th>\n",
       "      <td>50</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>140</td>\n",
       "      <td>233</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>163</td>\n",
       "      <td>0</td>\n",
       "      <td>0.6</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>71</th>\n",
       "      <td>51</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>94</td>\n",
       "      <td>227</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>154</td>\n",
       "      <td>1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>106</th>\n",
       "      <td>69</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>160</td>\n",
       "      <td>234</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>131</td>\n",
       "      <td>0</td>\n",
       "      <td>0.1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>270</th>\n",
       "      <td>46</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>120</td>\n",
       "      <td>249</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>144</td>\n",
       "      <td>0</td>\n",
       "      <td>0.8</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>102</th>\n",
       "      <td>63</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>140</td>\n",
       "      <td>195</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>179</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>242 rows × 13 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     age  sex  cp  trestbps  chol  fbs  restecg  thalach  exang  oldpeak  \\\n",
       "132   42    1   1       120   295    0        1      162      0      0.0   \n",
       "202   58    1   0       150   270    0        0      111      1      0.8   \n",
       "196   46    1   2       150   231    0        1      147      0      3.6   \n",
       "75    55    0   1       135   250    0        0      161      0      1.4   \n",
       "176   60    1   0       117   230    1        1      160      1      1.4   \n",
       "..   ...  ...  ..       ...   ...  ...      ...      ...    ...      ...   \n",
       "188   50    1   2       140   233    0        1      163      0      0.6   \n",
       "71    51    1   2        94   227    0        1      154      1      0.0   \n",
       "106   69    1   3       160   234    1        0      131      0      0.1   \n",
       "270   46    1   0       120   249    0        0      144      0      0.8   \n",
       "102   63    0   1       140   195    0        1      179      0      0.0   \n",
       "\n",
       "     slope  ca  thal  \n",
       "132      2   0     2  \n",
       "202      2   0     3  \n",
       "196      1   0     2  \n",
       "75       1   0     2  \n",
       "176      2   2     3  \n",
       "..     ...  ..   ...  \n",
       "188      1   1     3  \n",
       "71       2   1     3  \n",
       "106      1   1     2  \n",
       "270      2   0     3  \n",
       "102      2   2     2  \n",
       "\n",
       "[242 rows x 13 columns]"
      ]
     },
     "execution_count": 55,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(132    1\n",
       " 202    0\n",
       " 196    0\n",
       " 75     1\n",
       " 176    0\n",
       "       ..\n",
       " 188    0\n",
       " 71     1\n",
       " 106    1\n",
       " 270    0\n",
       " 102    1\n",
       " Name: target, Length: 242, dtype: int64, 242)"
      ]
     },
     "execution_count": 56,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_train, len(y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now we've got our data split into training and test sets, it's time to build a machine learning model.\n",
    "\n",
    "We'll train it (find the patterns) on the training set.\n",
    "\n",
    "And we'll test it (use the patterns) on the test set.\n",
    "\n",
    "We're going to try 3 different machine learning models:\n",
    "1. Logistic Regression \n",
    "2. K-Nearest Neighbours Classifier\n",
    "3. Random Forest Classifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Put models in a dictionary\n",
    "models = {\"Logistic Regression\": LogisticRegression(),\n",
    "          \"KNN\": KNeighborsClassifier(),\n",
    "          \"Random Forest\": RandomForestClassifier()}\n",
    "\n",
    "# Create a function to fit and score models\n",
    "def fit_and_score(models, X_train, X_test, y_train, y_test):\n",
    "    \"\"\"\n",
    "    Fits and evaluates given machine learning models.\n",
    "    models : a dict of differetn Scikit-Learn machine learning models\n",
    "    X_train : training data (no labels)\n",
    "    X_test : testing data (no labels)\n",
    "    y_train : training labels\n",
    "    y_test : test labels\n",
    "    \"\"\"\n",
    "    # Set random seed\n",
    "    np.random.seed(42)\n",
    "    # Make a dictionary to keep model scores\n",
    "    model_scores = {}\n",
    "    # Loop through models\n",
    "    for name, model in models.items():\n",
    "        # Fit the model to the data\n",
    "        model.fit(X_train, y_train)\n",
    "        # Evaluate the model and append its score to model_scores\n",
    "        model_scores[name] = model.score(X_test, y_test)\n",
    "    return model_scores"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/daniel/Desktop/ml-course/heart-disease-project/env/lib/python3.6/site-packages/sklearn/linear_model/_logistic.py:939: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html.\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "{'Logistic Regression': 0.8852459016393442,\n",
       " 'KNN': 0.6885245901639344,\n",
       " 'Random Forest': 0.8360655737704918}"
      ]
     },
     "execution_count": 58,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model_scores = fit_and_score(models=models,\n",
    "                             X_train=X_train,\n",
    "                             X_test=X_test,\n",
    "                             y_train=y_train,\n",
    "                             y_test=y_test)\n",
    "\n",
    "model_scores"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Model Comparison"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAXQAAAFPCAYAAABd3jU9AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjEsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy8QZhcZAAAZ1UlEQVR4nO3dfZRddX3v8feHBEwJD+Uh9QoBk1rwipAUiErBRUGUBT6AcgXhqpWUh1rhthUfilZsF2prbbWtitemLViryKVSlLYoJV6UWwUlCEoRoxSsTOm9RsCARTDB7/3jnMRxnGROyJnZmd95v9aaNWfvs+fMhxn95De/s/dvp6qQJM1+23UdQJI0HBa6JDXCQpekRljoktQIC12SGmGhS1Ij5nb1jffcc89atGhRV99ekmalm2+++btVtWCy5zor9EWLFrFq1aquvr0kzUpJ/m1TzznlIkmNsNAlqREWuiQ1orM5dI2udevWMTY2xiOPPNJ1lFlp3rx5LFy4kO23377rKNrGWOiacWNjY+y8884sWrSIJF3HmVWqivvuu4+xsTEWL17cdRxtY5xy0Yx75JFH2GOPPSzzxyEJe+yxh3/daFIWujphmT9+/uy0KRa6JDViZObQF53/j11HmFbfeucLuo7wuA37d7Mt/SzWr1/P3Lkj838zdcwRukbWi1/8Yg499FCe/vSns2LFCgA+/elPc8ghh7B06VKOOeYYAL7//e+zfPlyDjroIJYsWcIVV1wBwE477bTxtT7+8Y9z+umnA3D66adz3nnncfTRR/Pbv/3bfOlLX+Lwww/n4IMP5vDDD2f16tUAPPbYY7z+9a/f+Lrve9/7+MxnPsNLXvKSja977bXXctJJJ83Ej0MNcOigkXXxxRez++6784Mf/IBnPOMZnHjiiZx11llcf/31LF68mPvvvx+At73tbey6667cdtttADzwwANTvvY3vvENVq5cyZw5c3jwwQe5/vrrmTt3LitXruTNb34zV1xxBStWrODuu+/mlltuYe7cudx///3stttunHPOOaxZs4YFCxZwySWXsHz58mn9OagdFrpG1nvf+16uvPJKAO655x5WrFjBkUceufF0wN133x2AlStXctlll238ut12223K1z755JOZM2cOAGvXruVVr3oV3/zmN0nCunXrNr7uq1/96o1T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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "model_compare = pd.DataFrame(model_scores, index=[\"accuracy\"])\n",
    "model_compare.T.plot.bar();"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now we've got a baseline model... and we know a model's first predictions aren't always what we should based our next steps off. What should we do?\n",
    "\n",
    "Let's look at the following:\n",
    "* Hypyterparameter tuning\n",
    "* Feature importance\n",
    "* Confusion matrix\n",
    "* Cross-validation\n",
    "* Precision\n",
    "* Recall\n",
    "* F1 score\n",
    "* Classification report\n",
    "* ROC curve\n",
    "* Area under the curve (AUC)\n",
    "\n",
    "### Hyperparameter tuning (by hand)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Let's tune KNN\n",
    "\n",
    "train_scores = []\n",
    "test_scores = []\n",
    "\n",
    "# Create a list of differnt values for n_neighbors\n",
    "neighbors = range(1, 21)\n",
    "\n",
    "# Setup KNN instance\n",
    "knn = KNeighborsClassifier()\n",
    "\n",
    "# Loop through different n_neighbors\n",
    "for i in neighbors:\n",
    "    knn.set_params(n_neighbors=i)\n",
    "    \n",
    "    # Fit the algorithm\n",
    "    knn.fit(X_train, y_train)\n",
    "    \n",
    "    # Update the training scores list\n",
    "    train_scores.append(knn.score(X_train, y_train))\n",
    "    \n",
    "    # Update the test scores list\n",
    "    test_scores.append(knn.score(X_test, y_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[1.0,\n",
       " 0.8099173553719008,\n",
       " 0.7727272727272727,\n",
       " 0.743801652892562,\n",
       " 0.7603305785123967,\n",
       " 0.7520661157024794,\n",
       " 0.743801652892562,\n",
       " 0.7231404958677686,\n",
       " 0.71900826446281,\n",
       " 0.6942148760330579,\n",
       " 0.7272727272727273,\n",
       " 0.6983471074380165,\n",
       " 0.6900826446280992,\n",
       " 0.6942148760330579,\n",
       " 0.6859504132231405,\n",
       " 0.6735537190082644,\n",
       " 0.6859504132231405,\n",
       " 0.6652892561983471,\n",
       " 0.6818181818181818,\n",
       " 0.6694214876033058]"
      ]
     },
     "execution_count": 64,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_scores"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[0.6229508196721312,\n",
       " 0.639344262295082,\n",
       " 0.6557377049180327,\n",
       " 0.6721311475409836,\n",
       " 0.6885245901639344,\n",
       " 0.7213114754098361,\n",
       " 0.7049180327868853,\n",
       " 0.6885245901639344,\n",
       " 0.6885245901639344,\n",
       " 0.7049180327868853,\n",
       " 0.7540983606557377,\n",
       " 0.7377049180327869,\n",
       " 0.7377049180327869,\n",
       " 0.7377049180327869,\n",
       " 0.6885245901639344,\n",
       " 0.7213114754098361,\n",
       " 0.6885245901639344,\n",
       " 0.6885245901639344,\n",
       " 0.7049180327868853,\n",
       " 0.6557377049180327]"
      ]
     },
     "execution_count": 66,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_scores"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Maximum KNN score on the test data: 75.41%\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(neighbors, train_scores, label=\"Train score\")\n",
    "plt.plot(neighbors, test_scores, label=\"Test score\")\n",
    "plt.xticks(np.arange(1, 21, 1))\n",
    "plt.xlabel(\"Number of neighbors\")\n",
    "plt.ylabel(\"Model score\")\n",
    "plt.legend()\n",
    "\n",
    "print(f\"Maximum KNN score on the test data: {max(test_scores)*100:.2f}%\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Hyperparameter tuning with RandomizedSearchCV\n",
    "\n",
    "We're going to tune:\n",
    "* LogisticRegression()\n",
    "* RandomForestClassifier()\n",
    "\n",
    "... using RandomizedSearchCV"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create a hyperparameter grid for LogisticRegression\n",
    "log_reg_grid = {\"C\": np.logspace(-4, 4, 20),\n",
    "                \"solver\": [\"liblinear\"]}\n",
    "\n",
    "# Create a hyperparameter grid for RandomForestClassifier\n",
    "rf_grid = {\"n_estimators\": np.arange(10, 1000, 50),\n",
    "           \"max_depth\": [None, 3, 5, 10],\n",
    "           \"min_samples_split\": np.arange(2, 20, 2),\n",
    "           \"min_samples_leaf\": np.arange(1, 20, 2)}"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now we've got hyperparameter grids setup for each of our models, let's tune them using RandomizedSearchCV..."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Fitting 5 folds for each of 20 candidates, totalling 100 fits\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.\n",
      "[Parallel(n_jobs=1)]: Done 100 out of 100 | elapsed:    0.6s finished\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "RandomizedSearchCV(cv=5, error_score=nan,\n",
       "                   estimator=LogisticRegression(C=1.0, class_weight=None,\n",
       "                                                dual=False, fit_intercept=True,\n",
       "                                                intercept_scaling=1,\n",
       "                                                l1_ratio=None, max_iter=100,\n",
       "                                                multi_class='auto', n_jobs=None,\n",
       "                                                penalty='l2', random_state=None,\n",
       "                                                solver='lbfgs', tol=0.0001,\n",
       "                                                verbose=0, warm_start=False),\n",
       "                   iid='deprecated', n_iter=20, n_jobs=None,\n",
       "                   param_distributions={'C':...\n",
       "       4.83293024e-03, 1.27427499e-02, 3.35981829e-02, 8.85866790e-02,\n",
       "       2.33572147e-01, 6.15848211e-01, 1.62377674e+00, 4.28133240e+00,\n",
       "       1.12883789e+01, 2.97635144e+01, 7.84759970e+01, 2.06913808e+02,\n",
       "       5.45559478e+02, 1.43844989e+03, 3.79269019e+03, 1.00000000e+04]),\n",
       "                                        'solver': ['liblinear']},\n",
       "                   pre_dispatch='2*n_jobs', random_state=None, refit=True,\n",
       "                   return_train_score=False, scoring=None, verbose=True)"
      ]
     },
     "execution_count": 74,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Tune LogisticRegression\n",
    "\n",
    "np.random.seed(42)\n",
    "\n",
    "# Setup random hyperparameter search for LogisticRegression\n",
    "rs_log_reg = RandomizedSearchCV(LogisticRegression(),\n",
    "                                param_distributions=log_reg_grid,\n",
    "                                cv=5,\n",
    "                                n_iter=20,\n",
    "                                verbose=True)\n",
    "\n",
    "# Fit random hyperparameter search model for LogisticRegression\n",
    "rs_log_reg.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'solver': 'liblinear', 'C': 0.23357214690901212}"
      ]
     },
     "execution_count": 75,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "rs_log_reg.best_params_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8852459016393442"
      ]
     },
     "execution_count": 76,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "rs_log_reg.score(X_test, y_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now we've tuned LogisticRegression(), let's do the same for RandomForestClassifier()..."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Fitting 5 folds for each of 20 candidates, totalling 100 fits\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.\n",
      "[Parallel(n_jobs=1)]: Done 100 out of 100 | elapsed:  1.1min finished\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "RandomizedSearchCV(cv=5, error_score=nan,\n",
       "                   estimator=RandomForestClassifier(bootstrap=True,\n",
       "                                                    ccp_alpha=0.0,\n",
       "                                                    class_weight=None,\n",
       "                                                    criterion='gini',\n",
       "                                                    max_depth=None,\n",
       "                                                    max_features='auto',\n",
       "                                                    max_leaf_nodes=None,\n",
       "                                                    max_samples=None,\n",
       "                                                    min_impurity_decrease=0.0,\n",
       "                                                    min_impurity_split=None,\n",
       "                                                    min_samples_leaf=1,\n",
       "                                                    min_samples_split=2,\n",
       "                                                    min_weight_fraction_leaf=0.0,\n",
       "                                                    n_estimators=100,\n",
       "                                                    n_jobs...\n",
       "                   param_distributions={'max_depth': [None, 3, 5, 10],\n",
       "                                        'min_samples_leaf': array([ 1,  3,  5,  7,  9, 11, 13, 15, 17, 19]),\n",
       "                                        'min_samples_split': array([ 2,  4,  6,  8, 10, 12, 14, 16, 18]),\n",
       "                                        'n_estimators': array([ 10,  60, 110, 160, 210, 260, 310, 360, 410, 460, 510, 560, 610,\n",
       "       660, 710, 760, 810, 860, 910, 960])},\n",
       "                   pre_dispatch='2*n_jobs', random_state=None, refit=True,\n",
       "                   return_train_score=False, scoring=None, verbose=True)"
      ]
     },
     "execution_count": 79,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Setup random seed\n",
    "np.random.seed(42)\n",
    "\n",
    "# Setup random hyperparameter search for RandomForestClassifier\n",
    "rs_rf = RandomizedSearchCV(RandomForestClassifier(), \n",
    "                           param_distributions=rf_grid,\n",
    "                           cv=5,\n",
    "                           n_iter=20,\n",
    "                           verbose=True)\n",
    "\n",
    "# Fit random hyperparameter search model for RandomForestClassifier()\n",
    "rs_rf.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'n_estimators': 210,\n",
       " 'min_samples_split': 4,\n",
       " 'min_samples_leaf': 19,\n",
       " 'max_depth': 3}"
      ]
     },
     "execution_count": 80,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Find the best hyperparameters\n",
    "rs_rf.best_params_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8688524590163934"
      ]
     },
     "execution_count": 81,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Evaluate the randomized search RandomForestClassifier model\n",
    "rs_rf.score(X_test, y_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Hyperparamter Tuning with GridSearchCV\n",
    "\n",
    "Since our LogisticRegression model provides the best scores so far, we'll try and improve them again using GridSearchCV..."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Fitting 5 folds for each of 30 candidates, totalling 150 fits\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.\n",
      "[Parallel(n_jobs=1)]: Done 150 out of 150 | elapsed:    1.1s finished\n"
     ]
    }
   ],
   "source": [
    "# Different hyperparameters for our LogisticRegression model\n",
    "log_reg_grid = {\"C\": np.logspace(-4, 4, 30),\n",
    "                \"solver\": [\"liblinear\"]}\n",
    "\n",
    "# Setup grid hyperparameter search for LogisticRegression\n",
    "gs_log_reg = GridSearchCV(LogisticRegression(),\n",
    "                          param_grid=log_reg_grid,\n",
    "                          cv=5,\n",
    "                          verbose=True)\n",
    "\n",
    "# Fit grid hyperparameter search model\n",
    "gs_log_reg.fit(X_train, y_train);"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'C': 0.20433597178569418, 'solver': 'liblinear'}"
      ]
     },
     "execution_count": 84,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Check the best hyperparmaters\n",
    "gs_log_reg.best_params_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8852459016393442"
      ]
     },
     "execution_count": 85,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Evaluate the grid search LogisticRegression model\n",
    "gs_log_reg.score(X_test, y_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Evaluting our tuned machine learning classifier, beyond accuracy\n",
    "\n",
    "* ROC curve and AUC score\n",
    "* Confusion matrix\n",
    "* Classification report\n",
    "* Precision\n",
    "* Recall\n",
    "* F1-score\n",
    "\n",
    "... and it would be great if cross-validation was used where possible.\n",
    "\n",
    "To make comparisons and evaluate our trained model, first we need to make predictions."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Make predictions with tuned model\n",
    "y_preds = gs_log_reg.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0,\n",
       "       0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,\n",
       "       1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0])"
      ]
     },
     "execution_count": 88,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_preds"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "179    0\n",
       "228    0\n",
       "111    1\n",
       "246    0\n",
       "60     1\n",
       "      ..\n",
       "249    0\n",
       "104    1\n",
       "300    0\n",
       "193    0\n",
       "184    0\n",
       "Name: target, Length: 61, dtype: int64"
      ]
     },
     "execution_count": 89,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<sklearn.metrics._plot.roc_curve.RocCurveDisplay at 0x1a1fab0748>"
      ]
     },
     "execution_count": 90,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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YMCApNft3r37Ptd/5pln/NRTT/UOHTr4nj17qo1j/vz5/u1vfztlLHV18skn+2uvveZ79uzxQYMG+dy5c6uUGTt2rI8bN87d3VetWlVRzz179vgXX3zh7u47duzw4uJif/31193dfcGCBT5y5Mik10z2dwss9mo+V3XXkEiafj5nJSXrt9TrOQuOyeG273St9viLL75I06ZNueaaayr2tW/fnuuvvx6Ahx56iGeeeYZt27bx5Zdf8uCDD3LOOeewYsUKtm7dyogRIygpKaFLly5s3boVgE8++YSWLVty2GGHAXDYYYdVvH7//fe59tprKS8vp0WLFvz5z3+mc+fOzJkzh1/+8pfs2LGDVq1aMXXqVNq0acO4ceNYv349a9eupXXr1jzyyCPcfPPNzJs3DzPj6quvroj1vvvuY86cOezcuZPHH3+czp07M378eG699VY6d+4MwMEHH8wPfvADAC688EJmzpzJzTffDMCMGTMYPnx40t9Rr169OPjg/3ycTZ8+nRtvvJGJEyeycOHCKq2LZFLFsr/KysrYsmVLxfUvv/xynnrqKQYPHrxPuZKSkorWS+fOnVm7di0bNmygTZs2Ff82O3fuZOfOnRVjAKeddhpXXnklu3bt2qfu+0NdQyIHsJUrV9KrV6+UZV5//XUefvhhXnzxxX32T5w4kRYtWrB8+XJuvfVWlixZAkCPHj1o06YN+fn5jBgxgjlz5lS8Z9SoUdx3330sWbKEO++8s+KD8Jvf/CYLFy5k6dKlDBs2jPHjx1e8Z8mSJcyaNYtp06YxefJkPvjgA5YuXcry5cu55JJLKsq1bt2at956i9GjR3PnnXcCsGLFCnr37p20XsOHD2fGjBkAbN++nblz53L++edXKffqq6/uc46tW7fywgsvcM455zB8+HCmT5+e8ve3V6pYEs2fP5+ePXtW+Tn11FOrlF23bh25ubkV27m5uaxbt65KuR49evDEE08A8Oabb/Lhhx9SWloKwO7du+nZsydHH300AwcO5JRTTgHgoIMOomPHjrz99ttp1S8VtQhE0pTqm3umXHvttbzyyis0bdqURYsWATBw4ECOPPLIKmVffvllbrjhBgAKCwspLCwEoEmTJjz33HMsWrSIF154gZtuuoklS5YwduxYXnvtNS688MKKc2zfvh0Inqe4+OKLKSsrY8eOHfvcoz5kyBCaN28OwPPPP88111xT8Q01Ma7zzjsPgN69e1d86KVy8skn8+9//5t3332XVatW0adPH77+9a9XKVdWVkaXLl0qtp9++mn69+9PixYtOP/88/nFL37BXXfdRZMmTZLeUVPbu2z69+/PsmXL0irrScYDkl3vlltu4cYbb6Rnz550796dk046qeJ32KRJE5YtW8bmzZs599xzWbFiBd26dQPg6KOPZv369WklsFQiTQRmNgi4B2gC3O/ud1Q6figwBegNbAQudve1UcYk0pB07dqVv/3tbxXbEyZM4NNPP6WoqKhi39e+9rVq31/dh5yZUVxcTHFxMQMHDmTEiBGMGTOGI444IumH3PXXX8+YMWMYMmQICxYsYNy4cUmv7+7VXvPQQw8Fgg+2Xbt2VdRvyZIl9OjRI+l7hg0bxowZM1i1alXSbiGA5s2b73Pf/PTp03n11VfJy8sDYOPGjcyfP5+zzjqLVq1asWnTJlq3bg3AZ599VvG6plj2mj9/PjfddFOV/S1atOC1117bZ19ubm7FN3sIEuoxxxxT5b05OTn85S9/AYLfYX5+fpUHwo444gj69evHc889V5EItm3bVpGE6yKyriEzawJMAAYDBcBwMyuoVOwqYJO7dwTuAn4TVTwiDdGAAQPYtm0bEydOrNj31VdfpfXe008/nalTpwJBt8fy5cuB4M6it956q6LcsmXLaN++PTk5OeTn5/P4448DwQfS3m6Hzz//nGOPPRaAhx9+uNprfutb32LSpEkVH/SfffZZyhh/9KMf8etf/5r33nsPgD179vD73/++4vjw4cN59NFHefHFFxkyZEjSc3Tp0oXVq1cDsGXLFl555RU++ugj1q5dy9q1a5kwYUJF91C/fv145JFHgKDL5dFHH6V///5pxbLX3hZB5Z/KSQCgbdu2tGzZkoULF+LuTJkyhaFDh1Ypt3nzZnbs2AHA/fffz+mnn05OTg7l5eVs3rwZCLq8nn/++YoxDID33nuPrl3r3lKNcoygGFjt7mvcfQcwA6j8GxgK7P2r+itwpunpHZEKZsZTTz3FSy+9RH5+PsXFxVxxxRX85jc1f2caPXo0//73vyksLGT8+PEUFxcDwaDj2LFj6dy5Mz179mTmzJncc889AEydOpUHHnig4nbFvbeVjhs3jgsvvJDTTjut4ht0MiNHjqRdu3YUFhbSo0cPpk2b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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Plot ROC curve and calculate and calculate AUC metric\n",
    "plot_roc_curve(gs_log_reg, X_test, y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[25  4]\n",
      " [ 3 29]]\n"
     ]
    }
   ],
   "source": [
    "# Confusion matrix\n",
    "print(confusion_matrix(y_test, y_preds))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 96,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 216x216 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.set(font_scale=1.5)\n",
    "\n",
    "def plot_conf_mat(y_test, y_preds):\n",
    "    \"\"\"\n",
    "    Plots a nice looking confusion matrix using Seaborn's heatmap()\n",
    "    \"\"\"\n",
    "    fig, ax = plt.subplots(figsize=(3, 3))\n",
    "    ax = sns.heatmap(confusion_matrix(y_test, y_preds),\n",
    "                     annot=True,\n",
    "                     cbar=False)\n",
    "    plt.xlabel(\"True label\")\n",
    "    plt.ylabel(\"Predicted label\")\n",
    "    \n",
    "    bottom, top = ax.get_ylim()\n",
    "    ax.set_ylim(bottom + 0.5, top - 0.5)\n",
    "    \n",
    "plot_conf_mat(y_test, y_preds)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now we've got a ROC curve, an AUC metric and a confusion matrix, let's get a classification report as well as cross-validated precision, recall and f1-score."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 97,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.89      0.86      0.88        29\n",
      "           1       0.88      0.91      0.89        32\n",
      "\n",
      "    accuracy                           0.89        61\n",
      "   macro avg       0.89      0.88      0.88        61\n",
      "weighted avg       0.89      0.89      0.89        61\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(classification_report(y_test, y_preds))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Calculate evaluation metrics using cross-validation\n",
    "\n",
    "We're going to calculate accuracy, precision, recall and f1-score of our model using cross-validation and to do so we'll be using `cross_val_score()`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 98,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'C': 0.20433597178569418, 'solver': 'liblinear'}"
      ]
     },
     "execution_count": 98,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Check best hyperparameters\n",
    "gs_log_reg.best_params_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 99,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create a new classifier with best parameters\n",
    "clf = LogisticRegression(C=0.20433597178569418,\n",
    "                         solver=\"liblinear\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 100,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.81967213, 0.90163934, 0.86885246, 0.88333333, 0.75      ])"
      ]
     },
     "execution_count": 100,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Cross-validated accuracy\n",
    "cv_acc = cross_val_score(clf,\n",
    "                         X,\n",
    "                         y,\n",
    "                         cv=5,\n",
    "                         scoring=\"accuracy\")\n",
    "cv_acc"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 102,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8446994535519124"
      ]
     },
     "execution_count": 102,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "cv_acc = np.mean(cv_acc)\n",
    "cv_acc"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 103,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8207936507936507"
      ]
     },
     "execution_count": 103,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Cross-validated precision\n",
    "cv_precision = cross_val_score(clf,\n",
    "                         X,\n",
    "                         y,\n",
    "                         cv=5,\n",
    "                         scoring=\"precision\")\n",
    "cv_precision=np.mean(cv_precision)\n",
    "cv_precision"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 104,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9212121212121213"
      ]
     },
     "execution_count": 104,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Cross-validated recall\n",
    "cv_recall = cross_val_score(clf,\n",
    "                         X,\n",
    "                         y,\n",
    "                         cv=5,\n",
    "                         scoring=\"recall\")\n",
    "cv_recall = np.mean(cv_recall)\n",
    "cv_recall"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 105,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8673007976269721"
      ]
     },
     "execution_count": 105,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Cross-validated f1-score\n",
    "cv_f1 = cross_val_score(clf,\n",
    "                         X,\n",
    "                         y,\n",
    "                         cv=5,\n",
    "                         scoring=\"f1\")\n",
    "cv_f1 = np.mean(cv_f1)\n",
    "cv_f1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 107,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Visualize cross-validated metrics\n",
    "cv_metrics = pd.DataFrame({\"Accuracy\": cv_acc,\n",
    "                           \"Precision\": cv_precision,\n",
    "                           \"Recall\": cv_recall,\n",
    "                           \"F1\": cv_f1},\n",
    "                          index=[0])\n",
    "\n",
    "cv_metrics.T.plot.bar(title=\"Cross-validated classification metrics\",\n",
    "                      legend=False);"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Feature Importance\n",
    "\n",
    "Feature importance is another as asking, \"which features contributed most to the outcomes of the model and how did they contribute?\"\n",
    "\n",
    "Finding feature importance is different for each machine learning model. One way to find feature importance is to search for \"(MODEL NAME) feature importance\".\n",
    "\n",
    "Let's find the feature importance for our LogisticRegression model..."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 110,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Fit an instance of LogisticRegression\n",
    "clf = LogisticRegression(C=0.20433597178569418,\n",
    "                         solver=\"liblinear\")\n",
    "\n",
    "clf.fit(X_train, y_train);"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 111,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 0.00316727, -0.86044582,  0.66067073, -0.01156993, -0.00166374,\n",
       "         0.04386131,  0.31275787,  0.02459361, -0.60413038, -0.56862852,\n",
       "         0.45051617, -0.63609863, -0.67663375]])"
      ]
     },
     "execution_count": 111,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Check coef_\n",
    "clf.coef_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 114,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>sex</th>\n",
       "      <th>cp</th>\n",
       "      <th>trestbps</th>\n",
       "      <th>chol</th>\n",
       "      <th>fbs</th>\n",
       "      <th>restecg</th>\n",
       "      <th>thalach</th>\n",
       "      <th>exang</th>\n",
       "      <th>oldpeak</th>\n",
       "      <th>slope</th>\n",
       "      <th>ca</th>\n",
       "      <th>thal</th>\n",
       "      <th>target</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>63</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>145</td>\n",
       "      <td>233</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>150</td>\n",
       "      <td>0</td>\n",
       "      <td>2.3</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>37</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>130</td>\n",
       "      <td>250</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>187</td>\n",
       "      <td>0</td>\n",
       "      <td>3.5</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>41</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>130</td>\n",
       "      <td>204</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>172</td>\n",
       "      <td>0</td>\n",
       "      <td>1.4</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>56</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>120</td>\n",
       "      <td>236</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>178</td>\n",
       "      <td>0</td>\n",
       "      <td>0.8</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>57</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>120</td>\n",
       "      <td>354</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>163</td>\n",
       "      <td>1</td>\n",
       "      <td>0.6</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   age  sex  cp  trestbps  chol  fbs  restecg  thalach  exang  oldpeak  slope  \\\n",
       "0   63    1   3       145   233    1        0      150      0      2.3      0   \n",
       "1   37    1   2       130   250    0        1      187      0      3.5      0   \n",
       "2   41    0   1       130   204    0        0      172      0      1.4      2   \n",
       "3   56    1   1       120   236    0        1      178      0      0.8      2   \n",
       "4   57    0   0       120   354    0        1      163      1      0.6      2   \n",
       "\n",
       "   ca  thal  target  \n",
       "0   0     1       1  \n",
       "1   0     2       1  \n",
       "2   0     2       1  \n",
       "3   0     2       1  \n",
       "4   0     2       1  "
      ]
     },
     "execution_count": 114,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 113,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'age': 0.0031672721856887734,\n",
       " 'sex': -0.860445816920919,\n",
       " 'cp': 0.6606707303492849,\n",
       " 'trestbps': -0.011569930902919925,\n",
       " 'chol': -0.001663741604035976,\n",
       " 'fbs': 0.04386130751482091,\n",
       " 'restecg': 0.3127578715206996,\n",
       " 'thalach': 0.02459360818122666,\n",
       " 'exang': -0.6041303799858143,\n",
       " 'oldpeak': -0.5686285194546157,\n",
       " 'slope': 0.4505161679452401,\n",
       " 'ca': -0.6360986316921434,\n",
       " 'thal': -0.6766337521354281}"
      ]
     },
     "execution_count": 113,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Match coef's of features to columns\n",
    "feature_dict = dict(zip(df.columns, list(clf.coef_[0])))\n",
    "feature_dict"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 115,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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EhKCzMxMVKxYEefOncPOnTsl78GbCggIQI8ePdCjRw/07t0bpqam2L59O549ewZ/f38A796xJykmByoW7du3x48//oiVK1dixYoV0NPTw8cff4yVK1eidevWAPJ++a9evRoLFy7EihUrYGRkBDs7OyxatAjnzp3Dpk2b8OjRI1SoUAGffvopDh48iF9//RUnTpyAl5cXWrdujenTp2PTpk344YcfYGdnhxkzZiA2NhZRUVGvrWP9+vWxdetWLFu2DOvXr0dOTg7s7Owwd+5cdO/evZiPkJSrqys+/fRTrFixAqmpqXB3d0dAQIBaz7D847pixQosX74cBgYGcHFxwQ8//KD2gFlBNB3H1atXY+7cudi0aROEELCxscHkyZORk5ODH374ARcuXHjtFcmLatasiW3btmHRokVYu3Yt9PT04OzsjHnz5qmerXjXjj1JKYScXUKIqEjatGmDKlWqFNh7h0jXeM+BiIgkmByIiEiCyYGIiCR4z4GIiCR45UBERBJMDkREJFFinnN4/DgdSuWbt5CVL2+KpKS016+oJV3EKUn7UtLilKR9KWlxStK+FCWOnp4CFhYmBS4vMclBqRSFSg75r9EFXcQpSftS0uKUpH0paXFK0r7IHYfNSkREJMHkQEREEkwOREQkweRAREQSTA5ERCTB5EBERBJMDkREJFFinnP4EJiZl4ZxKc1vmZWVmcbyjMwcpD59XpzVIqISiMnhPWJcygBdAnYX6jV7A7sitZjqQ0QlF5uViIhIgsmBiIgkmByIiEiCyYGIiCSYHIiISILJgYiIJJgciIhIgsmBiIgkmByIiEiCyYGIiCSYHIiISILJgYiIJJgciIhIQtbksG/fPnz66adwdnZGx44dsWvXrleuv3v3btjb20v+Zs2aJWe1iIiokGQbsnv//v0YO3YsfH194enpiT/++AMTJkyAsbExOnTooPE1ly9fhq2tLebPn69WXqFCBbmqRURERSBbcli0aBE6duyIyZMnAwA8PT2RkpKCJUuWFJgcrly5gnr16sHV1VWuahARkQxkaVa6desWEhIS0K5dO7Xy9u3bIz4+Hrdu3dL4usuXL8Pe3l6OKhARkYxkSQ7x8fEAADs7O7VyW1tbAMD169clr3n48CGSkpLwzz//oEOHDqhXrx7at2//2vsURERU/GRpVkpNzZuI0tTUVK3cxMQE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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Visualize feature importance\n",
    "feature_df = pd.DataFrame(feature_dict, index=[0])\n",
    "feature_df.T.plot.bar(title=\"Feature Importance\", legend=False);"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 117,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th>target</th>\n",
       "      <th>0</th>\n",
       "      <th>1</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>sex</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>24</td>\n",
       "      <td>72</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>114</td>\n",
       "      <td>93</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "target    0   1\n",
       "sex            \n",
       "0        24  72\n",
       "1       114  93"
      ]
     },
     "execution_count": 117,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pd.crosstab(df[\"sex\"], df[\"target\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 119,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th>target</th>\n",
       "      <th>0</th>\n",
       "      <th>1</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>slope</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>12</td>\n",
       "      <td>9</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>91</td>\n",
       "      <td>49</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>35</td>\n",
       "      <td>107</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "target   0    1\n",
       "slope          \n",
       "0       12    9\n",
       "1       91   49\n",
       "2       35  107"
      ]
     },
     "execution_count": 119,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pd.crosstab(df[\"slope\"], df[\"target\"])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "slope - the slope of the peak exercise ST segment\n",
    "* 0: Upsloping: better heart rate with excercise (uncommon)\n",
    "* 1: Flatsloping: minimal change (typical healthy heart)\n",
    "* 2: Downslopins: signs of unhealthy heart"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Experimentation\n",
    "\n",
    "If you haven't hit your evaluation metric yet... ask yourself...\n",
    "\n",
    "* Could you collect more data?\n",
    "* Could you try a better model? Like CatBoost or XGBoost?\n",
    "* Could you improve the current models? (beyond what we've done so far)\n",
    "* If your model is good enough (you have hit your evaluation metric) how would you export it and share it with others?"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
