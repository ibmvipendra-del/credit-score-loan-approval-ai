
FEATURE_MAPPING = {
    "Attribute1": "checking_account_status",
    "Attribute2": "loan_duration_months",
    "Attribute3": "credit_history",
    "Attribute4": "loan_purpose",
    "Attribute5": "credit_amount",
    "Attribute6": "savings_account_status",
    "Attribute7": "employment_duration",
    "Attribute8": "installment_rate",
    "Attribute9": "personal_status_sex",
    "Attribute10": "other_debtors",
    "Attribute11": "residence_duration",
    "Attribute12": "property_type",
    "Attribute13": "age",
    "Attribute14": "other_installment_plans",
    "Attribute15": "housing_status",
    "Attribute16": "existing_credits",
    "Attribute17": "job_type",
    "Attribute18": "dependents",
    "Attribute19": "telephone",
    "Attribute20": "foreign_worker"
}

NUMERICAL_FEATURES = [
    "loan_duration_months",
    "credit_amount",
    "installment_rate",
    "residence_duration",
    "age",
    "existing_credits",
    "dependents"
]

CATEGORICAL_FEATURES = [
    feature for feature in FEATURE_MAPPING.values()
    if feature not in NUMERICAL_FEATURES
]

TARGET = "default"
