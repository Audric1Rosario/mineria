import time
from tkinter import ttk
import arff # pip install liac-arff
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB #pip install sklearn
#se cambio la línea 89 del archivo zoo.arff porque la librería daba error debido
#a que no reconocia el rango
#https://www.geeksforgeeks.org/naive-bayes-classifiers/#:~:text=Naive%20Bayes%20classifiers%20are%20a,is%20independent%20of%20each%20other.
from tkinter import *

class VentanaPrincipal:
    def __init__(self, modelador):
        self.modelador = modelador
        self.root = Tk()
        self.root.title("Clasificador de Animales")
        self.root.minsize(500, 400)
        self.root.resizable(False, False)
        style = ttk.Style(self.root)
        style.theme_use(style.theme_names()[0])
        nameFrame = Frame(self.root)#esto es puro visual
        nameFrame.grid(row=0, column=0, padx=10, pady=(10, 0), sticky=NSEW)
        nameFrame.grid_columnconfigure(0, weight=0)
        nameFrame.grid_columnconfigure(1, weight=1)
        Label(nameFrame, text="Nombre:").grid(row=0, column=0, sticky=W)
        Entry(nameFrame).grid(row=0, column=1, sticky=EW)
        self.opciones()
        self.resultado()
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(2, weight=1)
        self.root.mainloop()

    def queryResult(self):
        query = [i.get() for i in self.boolVars]
        query.insert(12, self.legCount.get())

        answer = self.modelador.predict([query])[0]
        self.answer.config(text=answer)

    def resultado(self):
        mainframe = Frame(self.root)
        mainframe.grid(row=2, column=0, padx=10, pady=(10, 0), sticky=NSEW)
        mainframe.grid_columnconfigure(0, weight=1)
        mainframe.grid_rowconfigure(1, weight=1)
        Button(mainframe, text="Clasificar", command=self.queryResult).grid(row=0, column=0, sticky=EW)
        resultFrame = LabelFrame(mainframe, text="Tipo:")
        resultFrame.grid(row=1, column=0, pady=(10, 0), sticky=NSEW)
        resultFrame.grid_columnconfigure(0, weight=1)
        resultFrame.grid_rowconfigure(1, weight=1)
        self.answer = Label(resultFrame, text="N/A")
        self.answer.grid(row=0,column=0, sticky=NSEW)
        self.answer.config(font=('Helvetica bold',40))

    def opciones(self):
        mainframe = LabelFrame(self.root, text="Rasgos")
        mainframe.grid(row=1, column=0, padx=10, pady=(10, 0), sticky=NSEW)
        mainframe.grid_columnconfigure(0, weight=1)
        legsFrame = Frame(mainframe)
        legsFrame.grid(row=0, column=0, sticky=EW)
        legsFrame.grid_columnconfigure(0, weight=0)
        legsFrame.grid_columnconfigure(1, weight=1)
        self.legCount = IntVar(value=0)

        optionMenu = OptionMenu(legsFrame, self.legCount, *range(0,10))
        optionMenu.grid(row=0, column=1, sticky=EW)
        Label(legsFrame, text="Cantidad de piernas:").grid(row=0, column=0, sticky=W)
        booleanFrame = Frame(mainframe)
        booleanFrame.grid(row=1, column=0, sticky=EW)
        booleanFrame.grid_columnconfigure(0, weight=1)
        booleanFrame.grid_columnconfigure(1, weight=1)
        booleanFrame.grid_columnconfigure(2, weight=1)
        self.boolVars = [IntVar(value=0) for _ in range(15)]
        attributosBooleanos = [
            "Pelo",
            "Plumas",
            "Huevos",
            "Leche",
            "Volador (airborne)",
            "Acuático",
            "Depredador",
            "Dientes",
            "Vertebrado",
            "Respira",
            "Venenoso",
            "Aletas",
            "Cola",
            "Domestico",
            "Tamaño de gato"
        ]
        for i in range(len(attributosBooleanos)):
            Checkbutton(booleanFrame, text=attributosBooleanos[i],
                        variable=self.boolVars[i],
                        onvalue=1, offvalue=0).grid(row=i%5, column=i//5, sticky=W)


def generateModel():
    print("Generando el modelo...")
    start_time = time.time()
    gnb = GaussianNB()
    with open('zoo.arff', 'r') as fOpen:
        dataset = arff.load(fOpen)
        data = []
        for j in dataset['data']:
            datos = []
            for i in j:
                if isinstance(i, str):
                    li = i.lower()
                    if li in ['true', 'false']:#transformando cada valor de booleano a 0 y 1, 1 como verdadero.
                        datos.append(1 if li == 'true' else 0)
                    else:
                        datos.append(i)
                else:
                    datos.append(i)
            data.append(datos)
        data = np.asarray(data)
        X = data[:, 1:-1]  # esta en 1 porque los nombres no significan nada
        # le = LabelEncoder()
        # X[:,0]=le.fit_transform(X[:,0])
        X = X.astype(int)# Hay cosas que no se transformo en int.
        y = data[:, -1]
        # train, set test_size there:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
        gnb.fit(X_train, y_train)
        # gnb.fit(X,y)
        print(f"El modelo fue generando en: {time.time() - start_time}s")
    return gnb

def main():
    VentanaPrincipal(generateModel())

if __name__ == '__main__':
    main()

