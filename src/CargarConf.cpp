/*
 * CargarConf.cpp
 *
 *  Created on: 14/08/2014
 *      Author: guillermo
 */


#include <fstream>
#include <iostream>
using namespace std;

int CargarConf()
{
    ifstream archivo("config.conf", ios::in);
    char linea[128];
    long contador = 0L;

    if(archivo.fail())
    cerr << "Error al abrir el archivo config.conf" << endl;
    else
    while(!archivo.eof())
    {
        archivo.getline(linea, sizeof(linea));
        cout << linea << endl;
        if((++contador % 24)==0)
        {
            cout << "CONTINUA...";
            cin.get();
        }
    }
    archivo.close();
    return 0;
}




