import sdv
import pandas as pd

df = pd.read_csv(r"C:\Users\110095\Desktop\HR_Data\HR_data_real.csv") 
print(df.shape) 
print(df.dtypes) 
print(df.isnull().sum())

from sdv.metadata import SingleTableMetadata

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df) 
metadata.visualize()

metadata.update_column(column_name='NUMBER_OF_PROMOTIONS', sdtype='numerical')
metadata.update_column(column_name='BASIC_PAY', sdtype='numerical')

from sdv.single_table import CTGANSynthesizer 
synthesizer = CTGANSynthesizer( 
    metadata, 
    enforce_min_max_values=True,
    enforce_rounding=True, 
    epochs=300,
    batch_size=500,
    generator_dim=(256, 256),
    discriminator_dim=(256, 256),
    verbose=True )
synthesizer.fit(df)

synthetic_df = synthesizer.sample(num_rows=10000) 
print(synthetic_df.shape)
synthetic_df.head()

df['_source'] = 'real' 
synthetic_df['_source'] = 'synthetic' 

combined_df = pd.concat([df, synthetic_df], ignore_index=True) 
combined_df = combined_df.drop(columns=['_source']) 
print(f"Total rows for ML training: {len(combined_df)}") 

from sdv.evaluation.single_table import run_diagnostic, evaluate_quality

diagnostic = run_diagnostic(
    real_data=df, 
    synthetic_data=synthetic_df, 
    metadata=metadata 
    ) 
diagnostic.get_results()

quality_report = evaluate_quality(
    real_data=df, 
    synthetic_data=synthetic_df, 
    metadata=metadata 
    ) 
quality_report.get_score()

import matplotlib.pyplot as plt 
import seaborn as sns 

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.histplot(df['MonthlyIncome'], ax=axes[0], label='Real', color='#1D9E75') 
sns.histplot(synthetic_df['MonthlyIncome'], ax=axes[0], label='Synthetic', color='#7F77DD', alpha=0.6) 
axes[0].legend() 
axes[0].set_title('MonthlyIncome distribution') 

df['Attrition'].value_counts(normalize=True).plot(kind='bar', ax=axes[1]) 
synthetic_df['Attrition'].value_counts(normalize=True).plot(kind='bar', ax=axes[1], alpha=0.6) 
axes[1].set_title('Attrition balance')

plt.tight_layout() 
plt.show()

from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import cross_val_score 

X_real = df.drop(columns=['Attrition']) 
y_real = df['Attrition'] 

X_comb = combined_df.drop(columns=['Attrition']) 
y_comb = combined_df['Attrition'] 

clf = RandomForestClassifier(n_estimators=100, random_state=42) 

real_score = cross_val_score(clf, X_real, y_real, cv=5).mean() 
comb_score = cross_val_score(clf, X_comb, y_comb, cv=5).mean() 

print(f"Real only accuracy: {real_score:.3f}") 
print(f"Real+Synthetic accuracy: {comb_score:.3f}") 

synthesizer.save('ctgan_hr_model.pkl')

from sdv.single_table import CTGANSynthesizer 

synthesizer = CTGANSynthesizer.load('ctgan_hr_model.pkl') 
synthetic_df = synthesizer.sample(num_rows=10000)

synthetic_df.to_csv('hr_synthetic_10k.csv', index=False)
combined_df.to_csv('hr_combined_15k.csv', index=False) 

print("Done! Files saved.")