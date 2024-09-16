import subprocess
import glob
import pandas as pd
import os


def main():
    aws_path = "s3://cat-veraset-demo/movement_premium/"
    out_path = "/glade/derecho/scratch/dgagne/veraset_cat/"
    dates = pd.date_range(start="2024-03-20", end="2024-03-31", freq="1D")
    for date in dates:
        print(date)
        date_path = date.strftime("%Y/%m/%d/") 
        day_path = os.path.join(out_path, date_path)
        if not os.path.exists(day_path):
            os.makedirs(day_path)
        subprocess.run(f"aws s3 cp {aws_path}{date_path} {day_path} --recursive", shell=True)
    return


if __name__ == "__main__":
    main()
