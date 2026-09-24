import numpy as np
import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import TVAESynthesizer

# 1. Load Data
file_path = r"C:\Users\110095\Desktop\HR_Data\masset_hr_fully_cleaned.csv"
data = pd.read_csv(file_path, encoding='utf-8', encoding_errors='replace')

# 2. Metadata Setup
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=data)
metadata.update_column(column_name='NATIVE_STATE', sdtype='categorical')
metadata.update_column(column_name='NUMBER_OF_PROMOTIONS', sdtype='numerical')

# 3. Define Constraints properly
all_constraints = []

# Range Constraints
cols_to_fix = [
    'NUMBER_OF_PROMOTIONS', 'APPROVED_LEAVES_PAST_3MONTHS',
    'REJECTED_LEAVES_PAST_3MONTHS', 'LOP_DAYS_PAST_3MONTHS',
    'TOTAL_LEAVES_PAST_3MONTHS','EXP_DAYS'
]

for col in cols_to_fix:
    all_constraints.append({
        'constraint_class': 'Range',
        'constraint_parameters': {'column_name': col, 'low': 0, 'strict_boundaries': False}
    })

all_constraints.append({
    'constraint_class': 'Range',
    'constraint_parameters': {'column_name': 'BASIC_PAY', 'low': 8000, 'high': 210000}
})

# Logic Constraints
all_constraints.append({
    'constraint_class': 'Inequality',
    'constraint_parameters': {'low_column_name': 'DATE_OF_BIRTH', 'high_column_name': 'JOINED_DATE'}
})

all_constraints.append({
    'constraint_class': 'Inequality',
    'constraint_parameters': {'low_column_name': 'BASIC_PAY', 'high_column_name': 'INHAND_SALARY'}
})

# Geographic Lock
all_constraints.append({
    'constraint_class': 'FixedCombinations',
    'constraint_parameters': {'column_names': ['NATIVE_STATE', 'NATIVE_DISTRICT']}
})

synthesizer = TVAESynthesizer( 
    metadata, 
    enforce_min_max_values=True,
    enforce_rounding=True, 
    epochs=500,
    batch_size=100,
    compress_dims=(128, 128),
    decompress_dims=(128, 128)
)

synthesizer.add_constraints(constraints=all_constraints)
synthesizer.fit(data)

synthetic_data = synthesizer.sample(num_rows=300)

synthetic_data['TOTAL_LEAVES_PAST_3MONTHS'] = (
    synthetic_data['LOP_DAYS_PAST_3MONTHS'] + 
    synthetic_data['APPROVED_LEAVES_PAST_3MONTHS']
)

synthetic_data.loc[synthetic_data['STATUS'] == 'Normal', 'DISCONTINUED_DATE'] = np.nan

# output_file = 'masset_hr_data_1.csv'
# synthetic_data.to_csv(output_file, index=False)

print("Success!!!")

from sdmetrics.reports.single_table import QualityReport

report = QualityReport()
report.generate(data, synthetic_data, metadata.to_dict())

print(f"Overall Score: {report.get_score()}")
column_shapes = report.get_details(property_name='Column Shapes')
print(column_shapes)