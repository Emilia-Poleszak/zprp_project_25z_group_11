## Temat projektu: Zreprodukowanie eksperymentów Hochreitera przy użyciu Linear Recurrent Unit (LRU)

### Wprowadzenie

W ramach projektu zaimplementowane zostaną 3 eksperymenty opisane 
w rozdziale 5 artykułu 
Long Short-Term Memory (Hochreiter & Schmidhuber, 1997) 
w architekturze LRU. Celem projektu jest przetestowanie skuteczności LRU
w opisanych zadaniach i porównanie jej z wynikami LSTM podanymi w artykule. 
Eksperymenty będą śledzone, a wnioski z nich będą opisane w raporcie badawczym.

### Planowana funkcjonalność programu

Program będzie wykorzystywał implementację LRU. Dla każdego eksperymentu 
odtworzona będzie architektura oraz parametry opisane w artykule LSTM 
z uwzględnieniem zamiany bloku LSTM na warstwę LRU. 
Utworzony generator danych będzie tworzył zbiory treningowe oraz testowe. 
Zapisywane będą wyniki eksperymentów zgodnie z założonymi kryteriami sukcesu 
w formie tabel i wykresów.

### Planowany zakres eksperymentów

1. Eksperyment 1: embedded reber grammar - standard benchmark test for recurrent nets (oznaczany później jako Reber)
2. Eksperyment 4: adding problem (oznaczany później jako Adding)
3. Eksperyment 5: multiplication problem (oznaczany później jako Multiplication)

### Planowany stack technologiczny

* Python 3.10 
* Biblioteki: pytorch, matplotlib, numpy, scipy, pandas
* LRU: https://github.com/Gothos/LRU-pytorch

### Harmonogram tygodniowy
1. 21-27.10.2025 - setup środowiska, przegląd materiałów pomocniczych, utworzenie generatora danych, generowanie danych
2. 28.10 - 03.11.2025 - sanity check wykorzystywanej implementacji LRU
3. 04-10.11.2025 - implementacja eksperymentu Reber
4. 11-17.11.2025 - pierwsze wyniki Reber, porównanie z wynikami z artykułu, dokumentacja wyników
5. 18-24.11.2025 - przerwa w pracy nad projektem na rzecz przygotowania do kolokwium
6. 25.11 - 01.12.2025 - implementacja eksperymentu Adding
7. 02-08.12.2025 - pierwsze wyniki Adding, porównanie z wynikami z artykułu, dokumentacja wyników
8. 09-15.12.2025 - implementacja eksperymentu Multiplication
9. 16-23.12.2025 - pierwsze wyniki Multiplication, porównanie z wynikami z artykułu, dokumentacja wyników
10. 24-29.12.2025 - przerwa świąteczna :)
11. 30.12.2025 - 05.01.2026 - implementacja generowania raportów do eksperymentów
12. 06-12.01.2026 - dokumentacja końcowa, porównanie całości wyników z wynikami z artykułu
13. 13-15.01.2026 - finalizacja projektu

15.01.2026 - termin końcowy

27.01.2026 - prezentacja

### Bibliografia

[1] https://www.researchgate.net/publication/13853244_Long_Short-Term_Memory

[2] https://github.com/Gothos/LRU-pytorch