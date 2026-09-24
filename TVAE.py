import pandas as pd
from sdv.constraints import FixedCombinations
from sdv.constraints import ScalarInequality, Constraint
from sdv.metadata import SingleTableMetadata
from sdv.single_table import TVAESynthesizer

file_path = r"C:\Users\110095\Desktop\Materials\MASSET MASTER.csv"

data = pd.read_csv(file_path, encoding='utf-8', encoding_errors='replace')

# Pre-process Dates
# date_cols = ['DATE_OF_BIRTH', 'JOINED_DATE']
# for col in date_cols:
#     if col in data.columns:
#         data[col] = pd.to_datetime(data[col], errors='coerce')

# Metadata
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=data)

# metadata.update_column(column_name='EMP_CODE',sdtype='id',regex_format='111[0-9]{3}')
# metadata.update_column(column_name='EMP_NAME', sdtype='categorical')
# metadata.update_column(column_name='DEPARTMENT', sdtype='categorical')
# metadata.update_column(column_name='DEPARTMENT_HEAD', sdtype='categorical')
metadata.update_column(column_name='NATIVE_STATE', sdtype='categorical')
metadata.update_column(column_name='EXP_DAYS', sdtype='numerical')
metadata.update_column(column_name='SALARY_INHAND', sdtype='numerical')
metadata.update_column(column_name='BASIC_PAY', sdtype='numerical')
metadata.update_column(column_name='NUMBER_OF_PROMOTIONS', sdtype='numerical')
metadata.update_column(column_name='APPROVED_LEAVES_PAST_3MONTHS', sdtype='numerical')
metadata.update_column(column_name='REJECTED_LEAVES_PAST_3MONTHS', sdtype='numerical')
metadata.update_column(column_name='LOP_DAYS_PAST_3MONTHS', sdtype='numerical')
metadata.update_column(column_name='TOTAL_LEAVES_PAST_3MONTHS', sdtype='numerical')

# if 'EMP_NAME' in data.columns:
#     metadata.update_column(column_name='EMP_NAME', sdtype='name')

# Constraints...
# non_negative_columns = [
#     'NUMBER_OF_PROMOTIONS', 
#     'APPROVED_LEAVES_PAST_3MONTHS', 
#     'REJECTED_LEAVES_PAST_3MONTHS', 
#     'LOP_DAYS_PAST_3MONTHS', 
#     'TOTAL_LEAVES_PAST_3MONTHS'
# ]

# leave_constraints = [
#     ScalarInequality(column_name=col, relation='>=', value=0) 
#     for col in non_negative_columns
# ]

class StatusDateConsistency(Constraint):
    """Ensure discontinued_date is null for Normal employees."""
    
    def is_valid(self, data):
        # Logic: If Status is 'Normal', Discontinued Date must be null
        # If Status is 'Resigned', Discontinued Date must NOT be null
        normal_mask = data['STATUS'] == 'Normal'
        resigned_mask = data['STATUS'] == 'Resigned'
        
        valid_normal = data.loc[normal_mask, 'DISCONTINUED_DATE'].isna()
        valid_resigned = data.loc[resigned_mask, 'DISCONTINUED_DATE'].notna()
        
        return valid_normal.all() and valid_resigned.all()

status_constraint = {
    'constraint_class': 'StatusDateConsistency',
    'column_names': ['STATUS', 'DISCONTINUED_DATE']
}

scalar_constraints = {
        'constraint_class': 'ScalarInequality',
        'constraint_parameters': {
            'column_name': 'NUMBER_OF_PROMOTIONS',
            'column_name': 'APPROVED_LEAVES_PAST_3MONTHS',
            'column_name': 'REJECTED_LEAVES_PAST_3MONTHS',
            'column_name': 'LOP_DAYS_PAST_3MONTHS',
            'column_name': 'TOTAL_LEAVES_PAST_3MONTHS',
            'relation': '>=',
            'value': 0
        }
}

total_vs_lop = {
    'constraint_class': 'Inequality',
    'constraint_parameters': {
        'low_column_name': 'LOP_DAYS_PAST_3MONTHS',
        'high_column_name': 'TOTAL_LEAVES_PAST_3MONTHS'
    }
}

total_vs_approved = {
    'constraint_class': 'Inequality',
    'constraint_parameters': {
        'low_column_name': 'APPROVED_LEAVES_PAST_3MONTHS',
        'high_column_name': 'TOTAL_LEAVES_PAST_3MONTHS'
    }
}

total_leaves_sum_logic = {
    'constraint_class': 'FormulaCustomConstraint',
    'constraint_parameters': {
        'column_names': [
            'TOTAL_LEAVES_PAST_3MONTHS', 
            'LOP_DAYS_PAST_3MONTHS', 
            'APPROVED_LEAVES_PAST_3MONTHS'
        ],
        'formula': '(TOTAL_LEAVES_PAST_3MONTHS >= (LOP_DAYS_PAST_3MONTHS + APPROVED_LEAVES_PAST_3MONTHS))'
    }
}

total_leaves_sum_logic = {
    'constraint_class': 'Sum',
    'constraint_parameters': {
        'column_names': ['LOP_DAYS_PAST_3MONTHS', 'APPROVED_LEAVES_PAST_3MONTHS'],
        'sum_column_name': 'TOTAL_LEAVES_PAST_3MONTHS',
        'relation': '>=' 
    }
}

# dept_head_constraint = {
#     'constraint_class': 'FixedCombinations',
#     'constraint_parameters': {
#         'column_names': ['DEPARTMENT', 'DEPARTMENT_HEAD']
#     }    
# }

birth_before_hire = {
    'constraint_class': 'Inequality',
    'constraint_parameters': {
        'low_column_name': 'DATE_OF_BIRTH',
        'high_column_name': 'JOINED_DATE'
    }
}

hire_before_resignation = {
    'constraint_class': 'Inequality',
    'constraint_parameters': {
        'low_column_name': 'JOINED_DATE',
        'high_column_name': 'DISCONTINUED_DATE'
    }
}

inhand_logic = {
    'constraint_class': 'Inequality',
    'constraint_parameters': {
        'low_column_name': 'BASIC_PAY',
        'high_column_name': 'INHAND_SALARY'
    }
}

age_logic = {
    'constraint_class': 'Inequality',
    'constraint_parameters': {
        'low_column_name': 'DATE_OF_BIRTH',
        'high_column_name': 'JOINED_DATE'
    }
}

leave_logic = {
    'constraint_class': 'Inequality',
    'constraint_parameters': {
        'low_column_name': 'APPROVED_LEAVES_PAST_3MONTHS',
        'high_column_name': 'TOTAL_LEAVES_PAST_3MONTHS'
    }
}

# Initializing
# synthesizer = TVAESynthesizer(
#     metadata,
#     enforce_min_max_values=True,
#     embedding_dim=256, 
#     compress_dims=(256, 256),
#     decompress_dims=(256, 256),
#     epochs=600
# )

synthesizer = TVAESynthesizer(
    metadata,
    enforce_min_max_values=True,
    epochs=600 
)

synthesizer.add_constraints(constraints=[birth_before_hire,scalar_constraints,
    age_logic,inhand_logic,hire_before_resignation,total_vs_lop,status_constraint,
    dept_head_constraint,leave_logic,total_leaves_sum_logic,total_vs_approved
])

print("Training TVAE model...")
synthesizer.fit(data)

synthetic_data = synthesizer.sample(num_rows=500)

output_file = 'asset_hr_data_generated_1.csv'
synthetic_data.to_csv(output_file, index=False)

print(f"Success! Data saved to {output_file}")
print(synthetic_data.head())

from sdv.evaluation.single_table import evaluate_quality
quality_report = evaluate_quality(
    real_data=data,
    synthetic_data=synthetic_data,
    metadata=metadata
)

print(f"Overall Quality Score: {quality_report.get_score() * 100:.2f}%")