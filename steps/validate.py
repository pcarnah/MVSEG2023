#!/usr/bin/env python
import argparse
import json
import tarfile


parser = argparse.ArgumentParser()
parser.add_argument("-r", "--results", required=True, help="validation results")
parser.add_argument("-e", "--entity_type", required=True, help="synapse entity type downloaded")
parser.add_argument("-s", "--submission_file", help="Submission File")

args = parser.parse_args()

if args.submission_file is None:
    prediction_file_status = "INVALID"
    invalid_reasons = ['Expected FileEntity type but found ' + args.entity_type]
else:
    invalid_reasons = []
    prediction_file_status = "VALIDATED"
    try:
        with tarfile.open(args.submission_file, "r") as tar_o:
            files = [f for f in tar_o.getnames() if f.endswith('nii.gz') or f.endswith('nii')]
        if not files:
            invalid_reasons.append("Submission must have Nifti files for each case with predicted labels")
            prediction_file_status = "INVALID"
    except:
        invalid_reasons.append("Could not open submission")
        prediction_file_status = "INVALID"
result = {'submission_errors': "\n".join(invalid_reasons),
          'submission_status': prediction_file_status}
with open(args.results, 'w') as o:
    o.write(json.dumps(result))