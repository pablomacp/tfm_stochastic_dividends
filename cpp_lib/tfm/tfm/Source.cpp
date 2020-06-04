// Primer programa

#include<iostream> // Paquete para el cout/cin
// #include<conio.h> // Función _getch() par aevitar que se cierre la consola. Mejor el pause de stdlib
#include<stdlib.h> // pause y numeros aleatorios
#include<time.h>

using namespace std;

int main(){

	cout << "Hola mundo" << endl;

	int numero, dato, contador = 0;

	srand(time(NULL));
	dato = 1 + rand() % (100);

	do{
		cout << "Inserta numero: "; cin >> numero;

		if (numero > dato){
			cout << "\nTe has pasado\n";
		}
		if (numero < dato){
			cout << "\nTe has quedado corto\n";
		}
		contador++;
	} while (numero != dato);

	system("pause");
	return 0;

}