from collections import Counter


def f_measure(hypothesis, reference):
    matches_total = 0
    h_length_total = 0
    r_length_total = 0

    for h_line, r_line in zip(hypothesis, reference):
        h_words = h_line.split()
        r_words = r_line.split()

        h_length_total += len(h_words)
        r_length_total += len(r_words)

        h_counter = Counter(h_words)
        r_counter = Counter(r_words)

        for word in h_counter:
            matches_total += min(r_counter[word], h_counter[word])
    if h_length_total == 0 and r_length_total == 0:
        return 1
    if h_length_total == 0 or r_length_total == 0:
        return 0
    return matches_total / ((h_length_total + r_length_total) / 2)


def rows_read(file_path):
    all_rows = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for row in file:
            all_rows.append(row)

    return all_rows


def main():
    ref_rows = rows_read('util/eval.ref')
    hyp1_rows = rows_read('util/eval.hyp1')
    hyp2_rows = rows_read('util/eval.hyp2')
    hyp3_rows = rows_read('util/eval.hyp3')

    f_measure1 = f_measure(hyp1_rows, ref_rows)
    f_measure2 = f_measure(hyp2_rows, ref_rows)
    f_measure3 = f_measure(hyp3_rows, ref_rows)

    print('F-Measure1:', f_measure1)
    print('F-Measure2:', f_measure2)
    print('F-Measure3:', f_measure3)


main()
