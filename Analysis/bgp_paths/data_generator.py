import random


def generate_random_numbers():
    """
    :return: A list of lists, where each list contains 15 random integers
    """
    randomlist = []
    for i in range(0, 100):
        n = random.sample(range(1, 10000), 15)
        randomlist.append(n)

    return randomlist

def save_data_to_txt(list):
    """
    :param list: Is a list of lists, containing integer numbers
    :return: A txt file containing the list of lists
    """
    with open('random_data.txt', 'w') as f:
        for line in list:
            f.write("%s\n" % line)

if __name__ == "__main__":

    randomlist = generate_random_numbers()
    save_data_to_txt(randomlist)
