import sys
import nltk

def main (file1,file2):
    pankt= nltk.data.load("nltk:tokenizers/punkt/english.pickle") #carico il tokenizzatore di NLTK
    letturafile1= open(file1,mode="r", encoding="utf-8") #mi apre il file1 in modalità di lettura con estensione utf-8
    letturafile2= open(file2, mode="r", encoding="utf-8") #mi apre il file2 in modalità di lettura con estensione utf-8
    letturatot = letturafile1.read() #legge il file1 
    letturatot2= letturafile2.read() #legge il file2
    bob= pankt.tokenize(letturatot)
    bob2= pankt.tokenize(letturatot2)
    lunghezza1 = numeroFrasi(bob)
    lunghezza2= numeroFrasi(bob2)
    print("-- NUMERO DI FRASI --")
    print("Il numero di frasi di", file1, "è di:", lunghezza1)
    print("Il numero di frasi di", file2, "è di", lunghezza2)
    print()
    lunghezzac1,tok1=numeroTokens(bob)
    lunghezzac2,tok2=numeroTokens(bob2)
    print("-- NUMERO DI TOKENS --")
    print("Il numero di tokens di", file1, "è di:", lunghezzac1)
    print("Il numero di tokens di", file2, "è di:", lunghezzac2)
    print()
    lunghezzaM1=lunghezzaMedia(lunghezzac1,lunghezza1)
    lunghezzaM2=lunghezzaMedia(lunghezzac2,lunghezza2)
    print("-- LUNGHEZZA MEDIA DELLE FRASI --")
    print("La lunghezza media delle frasi di", file1, "è di:", lunghezzaM1)
    print("La lunghezza media delle frasi di", file2, "è di:", lunghezzaM2)
    print()
    lunghezzaTokens(lunghezzac1,tok1)
    lunghezzaTokens(lunghezzac2,tok2)
    hapax1=numeroHapax(bob)
    hapax2=numeroHapax(bob2)
    print("-- NUMERO DI HAPAX --")
    print("Il numero di hapax sui primi 1000 tokens di", file1, "è di:", hapax1)
    print("Il numero di hapax sui primi 1000 tokens di", file2, "è di:", hapax2)
    print()
    grandezzaVocabolarioeTTR(tok1)
    grandezzaVocabolarioeTTR(tok2)
    PercentageAgg1,PercentageSost1, PercentageV1, PercentageAvv1, PercentageArt1,PercentagePrep1, PercentageCong1, PercentagePron1= distribuzione(bob)
    PercentageAgg2,PercentageSost2, PercentageV2, PercentageAvv2, PercentageArt2,PercentagePrep2, PercentageCong2, PercentagePron2= distribuzione(bob2)
    print()
    print("-- PERCENTUALE PAROLE PIENE E FUNZIONALI --")
    print()
    print(" :: PAROLE PIENE ::")
    print()
    print("1 --> La distribuzione in percentuale degli aggettivi in", file1, "è di:", PercentageAgg1, "%")
    print()
    print("2 --> La distribuzione in percentuale dei sostantivi in", file1, "è di:", PercentageSost1, "%")
    print()
    print("3 --> La distribuzione in percentuale dei verbi in", file1, "è di:", PercentageV1, "%")
    print()
    print("4 --> La distribuzione in percentuale degli avverbi in", file1, "è di:", PercentageAvv1, "%")
    print()
    print(" :: PAROLE FUNZIONALI ::")
    print()
    print("5 --> La distribuzione in percentuale degli articoli in", file1, "è di:", PercentageArt1, "%")
    print()
    print("6 --> La distribuzione in percentuale delle congiunzioni in", file1, "è di:", PercentageCong1, "%")
    print()
    print("7 --> La distribuzione in percentuale delle preposizioni in", file1, "è di:", PercentagePrep1, "%")
    print()
    print("8 --> La distribuzione in percentuale dei pronomi in", file1, "è di:", PercentagePron1, "%")
    print()
    print("----------------------------------------------------------------------------------------------")
    print(" :: PAROLE PIENE ::")
    print()
    print("1 --> La distribuzione in percentuale degli aggettivi in", file2, "è di:", PercentageAgg2, "%")
    print()
    print("2 --> La distribuzione in percentuale dei sostantivi in", file2, "è di:", PercentageSost2, "%")
    print()
    print("3 --> La distribuzione in percentuale dei verbi in", file2, "è di:", PercentageV2, "%")
    print()
    print("4 --> La distribuzione in percentuale degli avverbi in", file2, "è di:", PercentageAvv2, "%")
    print()
    print(" :: PAROLE FUNZIONALI ::")
    print()
    print("5 --> La distribuzione in percentuale degli articoli in", file2, "è di:", PercentageArt2, "%")
    print()
    print("6 --> La distribuzione in percentuale delle preposizioni in", file2, "è di:", PercentagePrep2, "%")
    print()
    print("7 --> La distribuzione in percentuale delle congiunzioni in", file2, "è di:", PercentageCong2, "%")
    print()
    print("8 --> La distribuzione in percentuale dei pronomi in", file2, "è di:", PercentagePron2, "%")


#Calcolo il numero di frasi    
def numeroFrasi(v): #per calcolare il numero di frasi basta utilizzare len che mi calcola la lunghezza delle frasi
    lenght= len(v)
    return lenght

#Calcolo il numero di Tokens 
def numeroTokens(s):
    numbertotale= [] #mi costruisco un array vuoto che conterrà il numero totale di tokens all'interno del corpus
    for frase in s: 
        number= nltk.word_tokenize(frase) #divido le frasi in token
        numbertotale= numbertotale+number #è il numero dei token
    lenght= len(numbertotale) #calcolo la lunghezza totale utilizzando len
    return lenght, numbertotale #restituisco il risultato richiamandolo nella funzione main

#Calcolo la lunghezza media delle frasi per tokens 
def lunghezzaMedia(s,v): #per calcolare la lunghezza media utilizzo la lunghezza delle frasi e il numero dei token
    media= s/v #basta dividere il numero dei token e quello delle frasi
    return media

#Calcolo la lunghezza media dei tokens senza punteggiatura
def lunghezzaTokens(l1,t1): #metto come parametri token (l1) e numero dei token (t1) (lista token)
    tokentot1= [] #lista dei miei token totali (conterrà il numero totale)
    lista= [",", ";", ":", ".", "!", "?"] #lista contenente dei caratteri che escluderò poi con not in
    count=0 #variabile che ha la funzione di contare i caratteri partendo da zero
    for token in t1: #token=frase e t1= lista token
        for car in token: #car=tokens e token=testo (nell'if chiedo che car=token non sia punteggiatura) 
            if car not in lista: #utilizzo un if per far sì che non mi venga presa in considerazione lista, nonché la punteggiatura
                count = count+1 #se ovviamente il carattere non è uguale a lista, di conseguenza aumento di 1
    media1= count/l1 #non è altro che il rapporto tra il numero totale dei caratteri e i token
    print("-- LUNGHEZZA MEDIA TOKENS --")
    print("La lunghezza media dei tokens escludendo la punteggiatura è di:\n")
    print(media1)

#calcolo il numero di hapax sui primi 1000 tokens
def numeroHapax(h1): #creo una funzione che calcoli gli hapax all'interno del testo che sappiamo essere i più presenti
    hapax=0 #variabile che mi conta gli hapax a partire da 0
    vocabolario = list(set(h1)) #calcolo il vocabolario
    count=0 #variabile che utilizzo per contare gli hapax
    for token in vocabolario: #scorro tutto il mio vocabolario
        count=count+1 #aumento sempre di 1
        frequenzaTok= h1.count(token) #conto
        if count <= 1000: #gli dico che count deve essere <= 1000 (primi 1000 tokens)
            if frequenzaTok==1: #essendo hapax, la frequenza deve essere uguale a 1
                hapax+=1 #allora aumento gli hapax di 1
    return hapax
    
#calcolo la  grandezzavocabolario e la TTR (incremento ogni 500)
def grandezzaVocabolarioeTTR (V): 
    n=500 #incrementa sempre di 500 unità dopo
    for i in range(0,len(V),n): #mi restituisce una lista e il parametro n mi scorre il testo in blocchi di 500
        tokens500=V[0:i+n] #voglio che il vocabolario cresca ogni 500, quindi 500/1000/1500, ...
        vocabolario500= list(set(tokens500)) 
        grandezzaV= len(vocabolario500) #per sapere la grandezza, basta che chieda di calcolare la lunghezza
        TTR = len(vocabolario500)/len(tokens500) #sappiamo essere il rapporto tra VOCABOLARIO (|V|) e CORPUS (|C|)
        print()
        print("-- GRANDEZZA VOCABOLARIO E TTR --")
        print()
        print("La grandezza del vocabolario e la ricchezza lessicale all'aumentare del corpus sono:\n")
        print("1 --> Incremento:", i,"-",i+500) #in questo modo incrementa di 500
        print("2 --> Grandezza Vocabolario:", grandezzaV)
        print("3 --> TTR:", TTR)

#distribuzione in termini di percentuale di parole piene e funzionali
def distribuzione(bob): 
    tokPoSTOTALI=[] #un array in cui ritroveremo le diverse POS (Part of Speech)
    Aggettivi=0 #inizializzo tutto a zero, poi all'interno dei vari if aumenterò sempre di 1 se è verificato
    Sostantivi=0
    Verbi=0
    Avverbi=0
    Articoli=0
    Preposizioni=0
    Congiunzioni=0
    Pronomi=0
    numberTokens=0 #variabile che mi serve per calcolare la percentuale
    for token in bob: #per ogni token presente in bob
        tokens= nltk.word_tokenize(token) 
        tokeninPoS= nltk.pos_tag(tokens) #utilizzo il pos_tag che prende in input una lista di tokens ed esegue l'analisi morfosintattica
        tokPoSTOTALI += tokeninPoS #POS totali
        numberTokens += len(tokens) #numero di tokens, quindi lunghezza del corpus
    for token in tokPoSTOTALI: #scorro i token e controllo le POS 
        if token[1] in ["JJ", "JJR", "JJS"]: #in questo caso se la POS [parola, POS] è un aggettivo, sost, ... allora incremento di 1
            Aggettivi+=1 #incremento aggettivi se l'if sopra è verificato
        if token[1] in ["NN", "NNS", "NNP", "NNPS"]:
            Sostantivi+=1 #incremento sostantivi se l'if è verificato
        if token[1] in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]:
            Verbi+=1 #incremento verbi se l'if è verificato
        if token[1] in ["RB", "RBR", "RBS"]:
            Avverbi+=1 #incremento avverbi se l'if è verificato
        if token[1] in ["DT"]:
            Articoli+=1 #incremento articoli se l'if è verificato
        if token[1] in ["IN"]:
            Preposizioni+=1 #incremento preposizioni se l'if è verificato
        if token[1] in ["CC", "IN"]:
            Congiunzioni+=1 #incremento congiunzioni se l'if è verificato
        if token[1] in ["WP", "WP$"]:
            Pronomi+=1 #incremento pronomi se l'if è verificato
    #Calcolo percentuale
    PercentageAgg= float(Aggettivi/numberTokens)*100
    PercentageSost= float(Sostantivi/numberTokens)*100
    PercentageV= float(Verbi/numberTokens)*100
    PercentageAvv= float(Avverbi/numberTokens)*100
    PercentageArt= float(Articoli/numberTokens)*100
    PercentagePrep= float(Preposizioni/numberTokens)*100
    PercentageCong= float(Congiunzioni/numberTokens)*100
    PercentagePron= float(Pronomi/numberTokens)*100

    return PercentageAgg,PercentageSost, PercentageV, PercentageAvv, PercentageArt, PercentagePrep, PercentageCong, PercentagePron


    
   




    
    



        
main (sys.argv[1], sys.argv[2])
