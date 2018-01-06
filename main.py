from hanaconnection import HanaConnection
import pandas as pd


def main():
    query = '''SELECT "GENDER", "ETHNICITY", "AGE", "HEIGHT", "WEIGHT", "BMI",
            "DIED_90DAYS", "LENGTH_OF_STAY_HOURS", "DOSAGE", "VENT_BEFORE",
            "VENT_AFTER", "AKIN", "ELIXHAUSER_VANWALRAVEN",
            "CONGESTIVE_HEART_FAILURE", "CARDIAC_ARRHYTHMIAS",
            "VALVULAR_DISEASE", "PULMONARY_CIRCULATION", "PERIPHERAL_VASCULAR",
            "HYPERTENSION", "PARALYSIS", "OTHER_NEUROLOGICAL",
            "CHRONIC_PULMONARY", "DIABETES_UNCOMPLICATED",
            "DIABETES_COMPLICATED", "HYPOTHYROIDISM", "RENAL_FAILURE",
            "LIVER_DISEASE", "PEPTIC_ULCER", "AIDS", "LYMPHOMA",
            "METASTATIC_CANCER", "SOLID_TUMOR", "RHEUMATOID_ARTHRITIS",
            "COAGULOPATHY", "OBESITY", "WEIGHT_LOSS", "FLUID_ELECTROLYTE",
            "BLOOD_LOSS_ANEMIA", "DEFICIENCY_ANEMIAS", "ALCOHOL_ABUSE",
            "DRUG_ABUSE", "PSYCHOSES", "DEPRESSION", "OASIS", "SOFA",
            "SOFA_RENAL","SAPS", "ANIONGAP", "ALBUMIN", "BANDS", "BICARBONATE",
            "BILIRUBIN", "CREATININE", "CHLORIDE", "GLUCOSE", "HEMATOCRIT",
            "HEMOGLOBIN", "LACTATE", "PLATELET", "POTASSIUM", "PTT", "INR",
            "PT", "SODIUM", "BUN", "WBC", "CR_24_B", "GFR_24_B", "CR_48_B",
            "GFR_48_B", "CR_72_B", "GFR_72_B", "CR_24_A", "GFR_24_A",
            "CR_48_A", "GFR_48_A", "CR_72_A", "GFR_72_A"
            FROM "MIMICIII"."M_RRT_DATA" '''
    result = None
    with HanaConnection() as conn:
        try:
            conn.execute(query)
            result = pd.DataFrame.from_records(conn.fetchall())
        except Exception as e:
            print(e)
    print(result)


if __name__ == '__main__':
    main()
