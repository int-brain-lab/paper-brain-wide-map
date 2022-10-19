import pandas as pd
from pathlib import Path
from one.api import ONE
from one.alf.exceptions import ALFError
from brainwidemap import bwm_query
from brainbox.io.one import SessionLoader

out_file = Path(__file__).parent.joinpath('check_wheel_data.csv')

one = ONE()
bwm_df = bwm_query(one, freeze='2022_10_update')

all_wheel = []
errors = []
for i, eid in enumerate(bwm_df['eid'].unique()):
    print(f"{i+1}/{bwm_df['eid'].nunique()}")

    wheel_checks = [eid]
    sess_loader = SessionLoader(one, eid)
    try:
        sess_loader.load_wheel()
        wheel_checks.append(True)  # Exists
        wheel_checks.append(True)  # Timestamps match
        wheel_checks.append(not sess_loader.wheel.empty)  # Not emtpy
        wheel_checks.append(set(sess_loader.wheel.columns) == set(['times', 'position', 'velocity', 'acceleration']))  # all columns
        wheel_checks.append(all(~sess_loader.wheel.isnull().values.all(axis=0)))  # not all nan for any column
    except ALFError as e:
        wheel_checks.append(False)  # Doesn't exist
        errors.append((eid, e))
    except ValueError as e:
        wheel_checks.append(True)  # Exist
        wheel_checks.append(False)  # Timestamps mismatch
        errors.append((eid, e))

    all_wheel.append(wheel_checks)

print(errors)
print(f'Saving csv to {out_file}')
wheel_df = pd.DataFrame(all_wheel, columns=['eid', 'exists', 'times_match', 'not_empty', 'all_columns', 'not_nan'])
wheel_df.to_csv(out_file)
