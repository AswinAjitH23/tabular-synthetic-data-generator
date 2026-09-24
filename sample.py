import numpy as np
import pandas as pd
from datetime import date
from sdv.constraints import FixedCombinations,Inequality,Constraint,Range
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer

file_path = r"C:\Users\110095\Desktop\HR_Data\masset_hr_fully_cleaned.csv"
data = pd.read_csv(file_path, encoding='utf-8', encoding_errors='replace')

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=data)
metadata.update_column(column_name='NATIVE_STATE', sdtype='categorical')
metadata.update_column(column_name='NUMBER_OF_PROMOTIONS', sdtype='numerical')

cols_to_fix = [
    'NUMBER_OF_PROMOTIONS', 'APPROVED_LEAVES_PAST_3MONTHS',
    'REJECTED_LEAVES_PAST_3MONTHS', 'LOP_DAYS_PAST_3MONTHS',
    'TOTAL_LEAVES_PAST_3MONTHS','EXP_DAYS'
]

numerical_constraints = []
for col in cols_to_fix:
    numerical_constraints.append({
        'constraint_class': 'Range',
        'constraint_parameters': {
            'column_name': col,
            'low': 0,
            'strict_boundaries': False
        }
    })

sum_constraint = {
    'constraint_class': 'PositiveSum',
    'constraint_parameters': {
        'column_names': ['LOP_DAYS_PAST_3MONTHS', 'APPROVED_LEAVES_PAST_3MONTHS'],
        'sum_title': 'TOTAL_LEAVES_PAST_3MONTHS'
    }
}

date_order_constraint = {
    'constraint_class': 'Inequality',
    'constraint_parameters': {
        'low_column_name': 'JOINED_DATE',
        'high_column_name': 'DISCONTINUED_DATE'
    }
}

all_constraints = numerical_constraints, sum_constraint, date_order_constraint

age_logic = {
    'constraint_class': 'Inequality',
    'constraint_parameters': {
        'low_column_name': 'DATE_OF_BIRTH',
        'high_column_name': 'JOINED_DATE'
    }
}

inhand_logic = {
    'constraint_class': 'Inequality',
    'constraint_parameters': {
        'low_column_name': 'BASIC_PAY',
        'high_column_name': 'INHAND_SALARY'
    }
}

lop_logic = {
    'constraint_class': 'Inequality',
    'constraint_parameters': {
        'low_column_name': 'LOP_DAYS_PAST_3MONTHS',
        'high_column_name': 'TOTAL_LEAVES_PAST_3MONTHS'
    }
}

basic_pay_constraint = {
    'constraint_class': 'Range',
    'constraint_parameters': {
        'column_name': 'BASIC_PAY',
        'low': 8000,
        'high': 210000
    }
}

state_district = {
    'constraint_class': 'FixedCombinations',
    'constraint_parameters': {
        'column_names': ['NATIVE_STATE', 'NATIVE_DISTRICT'] 
    }
}

all_constraints = []
all_constraints.extend(numerical_constraints)
all_constraints.append(date_order_constraint)
all_constraints.append(age_logic)
all_constraints.append(inhand_logic)
all_constraints.append(lop_logic)
all_constraints.append(sum_constraint)
all_constraints.append(basic_pay_constraint)
all_constraints.append(state_district)

synthesizer = CTGANSynthesizer( 
    metadata, 
    enforce_min_max_values=True,
    enforce_rounding=True, 
    epochs=300,
    batch_size=100,
    generator_dim=(256, 256),
    discriminator_dim=(256, 256),
    pac = 10,
    verbose=True 
    )

synthesizer.add_constraints(constraints=all_constraints)

synthesizer.fit(data)

synthetic_data = synthesizer.sample(num_rows=300)

synthetic_data.loc[synthetic_data['STATUS'] == 'Normal', 'DISCONTINUED_DATE'] = np.nan

synthetic_data['TOTAL_LEAVES_PAST_3MONTHS'] = (
    synthetic_data['LOP_DAYS_PAST_3MONTHS'] + 
    synthetic_data['APPROVED_LEAVES_PAST_3MONTHS']
)

#joined vs discontinued
j_date = pd.to_datetime(synthetic_data['JOINED_DATE'])
d_date = pd.to_datetime(synthetic_data['DISCONTINUED_DATE'])

mask = (j_date > d_date) & (synthetic_data['STATUS'] == 'Resigned')

synthetic_data.loc[mask, ['JOINED_DATE', 'DISCONTINUED_DATE']] = synthetic_data.loc[mask, ['DISCONTINUED_DATE', 'JOINED_DATE']].values

sys_date = pd.Timestamp.now().normalize()

# status vs discontinued
temp_joined = pd.to_datetime(synthetic_data['JOINED_DATE'])
temp_discontinued = pd.to_datetime(synthetic_data['DISCONTINUED_DATE'])

days_if_normal = (sys_date - temp_joined).dt.days
days_if_resigned = (temp_discontinued - temp_joined).dt.days

synthetic_data['EXP_DAYS'] = np.where(
    synthetic_data['STATUS'] == 'Normal',
    days_if_normal,
    days_if_resigned
)

synthetic_data['EXP_DAYS'] = synthetic_data['EXP_DAYS'].fillna(0).astype(int)

output_file = 'masset_hr_data_1.csv'
synthetic_data.to_csv(output_file, index=False)

print("Success!!!")

from sdmetrics.reports.single_table import QualityReport

report = QualityReport()
report.generate(data, synthetic_data, metadata.to_dict())

print(f"Overall Score: {report.get_score()}")
column_shapes = report.get_details(property_name='Column Shapes')
print(column_shapes)