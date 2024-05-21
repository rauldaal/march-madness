from multiprocessing import Process


def func1():
    print("func1: starting")
    for i in range(10000000):
        pass

    print("func1: finishing")


def func2():
    print("func2: starting")
    for i in range(10):
        pass

    print("func2: finishing")


if __name__ == "__main__":
    p1 = Process(target=func1)
    p1.start()
    p2 = Process(target=func2)
    p2.start()
    p1.join()
    p2.join()