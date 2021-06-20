import multiprocessing, subprocess


def worker():
    subprocess.Popen(['C:\Dev\Tool\Tool\ML_otricatelni_-_kubichni.pyw'], shell=True)

if __name__ == '__main__':
    
    for i in range(3):
        p = multiprocessing.Process(target=worker)
        p.start()