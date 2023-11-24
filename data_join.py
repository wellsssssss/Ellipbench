import csv
import glob

def concatenate_csv(input_prefix, output_file):
    with open(output_file, 'w', newline='') as output:
        writer = csv.writer(output)

        # 获取所有以指定前缀开头的CSV文件
        files = glob.glob(f"{input_prefix}_*.csv")

        # 逐个读取CSV文件并写入输出文件
        for file in files:
            with open(file, 'r') as input_file:
                reader = csv.reader(input_file)
                for row in reader:
                    writer.writerow(row)

    print(f"成功将{len(files)}个CSV文件拼接成一个文件。")

# 使用示例
input_prefix = "csv1"  # 输入CSV文件名前缀（与之前分批写入的保持一致）
output_file = "combined.csv"  # 输出CSV文件名

concatenate_csv(input_prefix, output_file)