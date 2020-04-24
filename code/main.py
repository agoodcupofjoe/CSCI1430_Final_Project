def main():
    dictionary = {}
    file = open("../data/test_dictionary.txt","r")
    for line in file:
        dictionary[line.strip()] = True

    while True:
        user_input = input("Enter line: ")
        words = user_input.split()
        tokens = [w for w in words if w in dictionary]
        print(tokens)

if __name__ == '__main__':
    main()