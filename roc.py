import matplotlib.pyplot as plt

y = [0,2/6,4/6,6/6,6/6]
x = [0,0,1/4,2/4,4/4]
def main():#Just draws the graph for q1_5.
    plt.plot(x,y)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("q5a")
    plt.show()

if __name__ == "__main__":
    main()