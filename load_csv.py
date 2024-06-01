from csv import reader

# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
    for row in csv_reader:
        if not row:
            continue
        dataset.append(row)
    return dataset