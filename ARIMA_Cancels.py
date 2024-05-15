from collections import defaultdict
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def print_cancellations(filename):
    weekly_cancellations = defaultdict(int)
    with open(filename, 'r') as file:
        next(file)  # Skip the header
        for line in file:
            fields = line.strip().split(',')
            if fields[0] == "1":  # Check if "IsCanceled" is 1
                year = int(fields[2])
                month = datetime.strptime(fields[3], '%B').month  # Convert month name to number
                day = int(fields[5])
                date = datetime(year, month, day)
                week_start = date - timedelta(days=(date.weekday() + 1) % 7)  # Sunday
                weekly_cancellations[week_start] += 1

    print("\nWeekly aggregated cancellations:")
    dates = []
    counts = []
    for week_start, count in sorted(weekly_cancellations.items()):
        print(f'{week_start.date()} {count}')
        dates.append(week_start.date())
        counts.append(count)

    # Plotting
    plt.plot(dates, counts)
    plt.title('Weekly Cancellations')
    plt.xlabel('Date')
    plt.ylabel('Number of Cancellations')
    plt.show()

print_cancellations("H1.csv")

