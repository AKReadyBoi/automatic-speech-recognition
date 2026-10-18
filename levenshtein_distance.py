import numpy as np


def lev_dist(hypothesis, reference):
    matrix = np.zeros((len(reference) + 1, len(hypothesis) + 1), dtype=int)
    for i in range(len(hypothesis) + 1):
        matrix[0, i] = i
    for i in range(len(reference) + 1):
        matrix[i, 0] = i
    matrix[0, 0] = 0

    for i in range(len(reference)):
        for j in range(len(hypothesis)):
            if reference[i] == hypothesis[j]:
                matrix[i + 1, j + 1] = min(matrix[i, j], matrix[i + 1, j] + 1, matrix[i, j + 1] + 1)
            else:
                matrix[i + 1, j + 1] = min(matrix[i, j] + 1, matrix[i + 1, j] + 1, matrix[i, j + 1] + 1)
    lev_distance = matrix[len(reference)][len(hypothesis)]
    j = len(reference)
    k = len(hypothesis)
    operations = ''
    while j > 0 or k > 0:
        if j > 0 and k > 0:
            cost = 0 if reference[j - 1] == hypothesis[k - 1] else 1
            if matrix[j][k] == matrix[j - 1][k - 1] + cost:
                if cost == 1:
                    operations = operations + 's'
                else:
                    operations = operations + 'm'
                j -= 1
                k -= 1
                continue
        if j > 0 and matrix[j][k] == matrix[j - 1][k] + 1:
            operations = operations + 'i'
            j -= 1
        elif k > 0 and matrix[j][k] == matrix[j][k - 1] + 1:
            operations = operations + 'd'
            k -= 1

    #print(operations[::-1])
    return lev_distance


def rows_read(file_path):
    all_rows = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for row in file:
            all_rows.append(row)

    return all_rows


def calculate_wer(hypothesis_rows, reference_rows):
    total_distance = 0
    total_ref_words = 0

    for hyp_line, ref_line in zip(hypothesis_rows, reference_rows):
        hyp_words = hyp_line.split()
        ref_words = ref_line.split()
        total_distance += lev_dist(hyp_words, ref_words)
        total_ref_words += len(ref_words)

    if total_ref_words == 0:
        return 0.0
    return total_distance / total_ref_words


def main():
    ref_rows = rows_read('util/eval.ref')
    if ref_rows:
        for i in range(1, 4):
            hyp_file = f'util/eval.hyp{i}'
            hyp_rows = rows_read(hyp_file)
            if hyp_rows:
                wer = calculate_wer(hyp_rows, ref_rows)
                print(f"WER {i} ({hyp_file}): {wer:.4f}")


if __name__ == "__main__":
    main()
