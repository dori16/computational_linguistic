import sys
import nltk
from nltk import trigrams, bigrams
import math
def ATesto(frasi):
    #lunghezza totale del testo
    lunghezzaTotale=0.0
    #lista dei token 
    listaToken = []
    #lista delle POS
    tokenPOSTot = []
    for frase in frasi:
        #divido in token la frase
        token = nltk.word_tokenize(frase)
        #lista con tutti i token del testo
        tokenPOS = nltk.pos_tag(token)
        #Calcolo tutti i token del testo
        listaToken = listaToken + token
        #calcolo tutti i POS del testo
        tokenPOSTot = tokenPOSTot + tokenPOS
        lunghezzaTotale = lunghezzaTotale + len(token) 
    #restituisco la lunghezza totale del testo e la lunghezza dei token 
    return lunghezzaTotale, listaToken, tokenPOSTot



#estrarre e ordinare in ordine di frequenza decrescente, indicando anche la relativa frequenza:
#le 10 PoS più frequenti
#i 10 bigrammi di PoS più frequenti
#i 10 trigrammi di PoS più frequenti
#i 20 aggettivi e i 20 avverbi più frequenti


#funzione che calcola i 10 POS piu' frequenti
def PoSfreq(tokenPOSTot):
    #lista che contiene le POS
    PartOfSpeech = []
    #scorro le POS le aggiungo alla lista definita precedentemente
    for token in tokenPOSTot:
        #calcolo la distribuzione di frequenza delle POS e le 10 POS piu' frequenti 
        PartOfSpeech.append(token[1])
    #calcolo la distribuzione di frequenza delle POS e le 10 POS piu' frequenti
    DistribuzionePOS = nltk.FreqDist(PartOfSpeech)
    POSFrequenti = DistribuzionePOS.most_common(10)
    return POSFrequenti


#funzione che stampa le distribuzioni di frequenza 
def frequenze(Distribuzione):
    for elem in Distribuzione:
        print ("\t", elem[0], "compare", elem[1], "volte")

def stampafreq(Dist):
    for elem in Dist:
        print ("\t", elem[0], "compare", elem[1], "volte")

#funzione che calcola la frequenza de trigrammi
def Calcolabigrammi(bigrammi):
    #calcolo la frequenza dei trigrammi di POS
    Distribuzionebigrammi = nltk.FreqDist(bigrammi)
    #estraggo i 10 trigrammi di POS più frequenti
    bigrammiFrequenti = Distribuzionebigrammi.most_common(10)
    return bigrammiFrequenti
#funzione che stampa le frequenze dei bigrammi
def frequenzabigrammi(bigrammifreq):
    for elem in bigrammifreq:
        print ("\t", elem[0][0][1], elem[0][1][1],  "compare", elem[1], "volte")


#funzione che calcola la frequenza de trigrammi
def Calcolatrigrammi(trigrammi):
    #calcolo la frequenza dei trigrammi di POS
    DistribuzioneTrigrammi = nltk.FreqDist(trigrammi)
    #estraggo i 10 trigrammi di POS più frequenti
    trigrammiFrequenti = DistribuzioneTrigrammi.most_common(10)
    return trigrammiFrequenti

#funzione che stampa le frequenze dei trigrammi
def frequenzatrigrammi(trigrammifreq):
    for elem in trigrammifreq:
        print ("\t", elem[0][0][1], elem[0][1][1], elem[0][2][1], "compare", elem[1], "volte")

#funzione che calcola i 20 aggettivi piu' frequenti
def aggettivi(tokenPOSTot):
    #lista dgli aggettivi frequenti
    AggFreq = []
    #Scorro le POS token per token
    for token in tokenPOSTot:
        #se il token è un aggettivo lo aggiungo alla lista degli aggettivi frquenti
        if token[1] in {"JJ", "JJR", "JJS"}:
            AggFreq.append(token[0])
    #calcolo la distribuzione di frequenza degli aggettivi e i 20 aggettivi più frequenti
    DistribuzioneAggettivi = nltk.FreqDist(AggFreq)
    aggettivifrequenti = DistribuzioneAggettivi.most_common(20)
    return aggettivifrequenti


#funzione che calcola i 20 avverbi piu' frequenti
def avverbi(tokenPOSTot):
    #lista degli avverbi frequenti
    aFreq = []
    #Scorro le POS token per token
    for token in tokenPOSTot:
        #se il token è un avverbio lo aggiungo alla lista degli avverbi frequenti
        if token[1] in {"RB", "RBS", "RBR", "WRB"}:
            aFreq.append(token[0])
    #calcolo la distribuzione d frequenza degli avverbi e i 20 avverbi più frequenti
    Distribuzioneavverbi = nltk.FreqDist(aFreq)
    avverbifrequenti = Distribuzioneavverbi.most_common(20)
    return avverbifrequenti




    
"""estraete ed ordinate i 20 bigrammi di token composti da aggettivo e sostantivo (dove ogni token deve avere una frequenza
maggiore di 3):
◦ con frequenza massima, indicando anche la relativa frequenza;
◦ con probabilità condizionata massima, indicando anche la relativa probabilità;
◦ con forza associativa (calcolata in termini di Local Mutual Information) massima,
indicando anche la relativa forza associativa;"""


def listaSostagg(tokenPOSTot,listaToken):
    lista=[] #lista che contiene i miei bigrammi, quindi la sequenza: aggettivo, sostantivo
    listaLMI=[] #lista della LMI cosituita da: (bigramma e LMI)
    ListaPCond= [] #lista Probabilità condizionata che sarà costituita da: bigramma, prob. condizionata
    bigrammiTokens= list(bigrams(listaToken)) 
    listab= list(bigrams(tokenPOSTot))  
    for token in listab: 
        if (token[0][1] in ["JJ", "JJR", "JJS"]) and (token[1][1] in ["NNP", "NNPS", "NN", "NNS"]): 
            frequenza1= listaToken.count(token[0][0]) 
            frequenza2= listaToken.count(token[1][0]) 
            if frequenza1 > 3 and frequenza2 > 3: 
                lista.append((token[0][0], token[1][0])) 
    bigrammii= nltk.FreqDist(lista) #utilizzo la distribuzione di frequenza considerando lista
    bigramsoggagg= bigrammii.most_common(20) #voglio i 20 più frequenti
    bigrammiDiversi= list(set(lista)) #tolgo le ripetizioni
    for token in bigrammiDiversi:
        frequBigram= bigrammiTokens.count(token) #frequenza bigrammi (che calcolo andando a contare)
        frequenzaElemento1= listaToken.count(token[0]) #frequenza primo elemento bigram
        frequenzaElemento2= listaToken.count(token[1]) #frequenza secondo elemento bigram
        ProbCondizionata= frequBigram/frequenzaElemento1 #la probabilità condizionata è P(A,B)/P(B)
        ListaPCond.append((token, ProbCondizionata)) #la aggiungo all'interno della lista 
        probabilitàtoken1= frequenzaElemento1/len(listaToken) #frequenza dell'elemento e corpus
        probabilitàtoken2= frequenzaElemento2/len(listaToken) #frequenza dell'altro elemento e corpus
        ProbCongiunta= ProbCondizionata*probabilitàtoken1 #faccio riferimento alla regola del prodotto P(A,B)=P(A)*P(B)
        Probabilità= ProbCongiunta/(probabilitàtoken1*probabilitàtoken2) #la calcolo per calcolare la MI
        MI= math.log(Probabilità,2) #gli dico che il logaritmo è in base due e che prenda in considerazione la variabile probabilità
        LMI= frequBigram*MI #sappiamo essere la MI*frequenzaosservata, quindi p(<u,v>)*MI
        listaLMI.append((token, LMI)) #la aggiungo all'interno della lista 
    Ordinamento= sorted(ListaPCond, reverse= True, key= lambda x:x[1]) #ordinamento decrescente sulla base della p. condizionata
    Pcond= Ordinamento[:20] #la soglia che non deve superare è 20 e ottengo i 20 bigrammi (bigramma, P.Cond)
    OrdinamentoLMI= sorted(listaLMI, reverse=True, key= lambda x:x[1]) #ordinamento decrescente per la LMI
    lmi= OrdinamentoLMI[:20] #ottengo i 20 bigrammi (bigramma, LMI)
    
 
    return lmi, Pcond, bigramsoggagg


#estrarre le frasi con 6<tokens<25 che occorre almeno due volte nel corpus di riferimento

def almenodue(frasi, listaToken):
    #inizializzo la lista che contiene le frasi
    listaFrasi = []
    #scorro la lista una frase alla volta
    for frase in frasi:
        #tokenizzo la frase
        tokens = nltk.word_tokenize(frase)
        #controllo se la lunghezza della frase in termini di token corrisponde ad un numero compreso tra 6 e 25
        if len(tokens)>6 and len(tokens)<25:
            #scorro la lista un token alla volta
            for tok in tokens:
                #calcolo la frequenza del token
                freqTok = listaToken.count(tok)
                #controllo se ogni token ha frequenza maggiore di 2
            if freqTok>2:
                    #aggiungo la frase alla lista
                     
             listaFrasi.append(frase)
    #restituisco i risultati
    return listaFrasi

def Calcola0(frase, distribuzioneTok, numeroToken):
    #scorro la lista un token alla volta
    for tok in frase:
        #calcolo la probabilita del token con la distribuzione di frequenza
        probTok = (float(distribuzioneTok[tok]))/(numeroToken)
    #restituisco il risultato
    return probTok



def CalcolaMarkov2(frase, distribuzioneTok, numeroToken, bigrammiTokPos, probTok):
    #calcolo la distribuzione di frequenza dei bigrammi
    distribuzioneBig = nltk.FreqDist(bigrammiTokPos)
    #scorro la lista un token alla volta
    for tok in frase:
        #calcolo la probabilita del token con la distribuzione di frequenza
         
        #divido la frase in bigrammi
        bigrammiFrase = list(bigrams(frase))
            #scorro la lista un bigramma alla volta
        for bigramma in bigrammiFrase:
                #calcolo la probabilita del bigramma con la distribuzione di frequenza
                probBig = (float(distribuzioneBig[bigramma]))/(numeroToken)
                probig = (float(distribuzioneBig[bigramma[0]]))/(numeroToken)
                #calcolo la probabilita della frase tramite un modello di Markov di ordine 2
                probabilita = (float(probBig)/(probTok)) * (float(probig)/(probTok))
    #restituisco il risultato
    return probabilita




def CalcolaProbFrasi(listaFrasi, distribuzioneTok, numeroToken, bigrammiTokPos, probTok):
    #inizializzo le variabili per il calcolo delle probabilita delle frasi
    probMax0 = 0
    probMax1 = 0
    fraseProbMax0 = ""
    fraseProbMax1 = ""
    #scorro la lista una frase alla volta
    for frase0 in listaFrasi:
        #calcolo la probabilita della frase con un modello di Markov di ordine 0
        prob0 = Calcola0(frase0, distribuzioneTok, numeroToken)
        #controllo se la frase ha probabilita maggiore di probMax0
        if prob0>probMax0:
            #assegno alla variabile probMax0 il valore della probabilita della frase
            probMax0 = prob0
            #assegno alla variabile fraseProbMax0 la frase con probabilita maggiore
            fraseProbMax0 = frase0
    #stampo la frase con probabilita maggiore 
    print ("Frase: '", fraseProbMax0, "' Probabilità:", probMax0, ".")
    print ("\n")
    print ("\n")
#calcolo la probabilita con un modello di Markov di ordine 2
    for frase1 in listaFrasi:

        prob1 = CalcolaMarkov2(frase1, distribuzioneTok, numeroToken, bigrammiTokPos, probTok)
        #controllo se la frase ha probabilita massima
        if prob1>probMax1:
                #assegno alla variabile probMax1 il valore della probabilita della frase
           probMax1 = prob1
                  #assegno alla variabile fraseProbMax1 la frase con probabilita maggiore
           fraseProbMax1 = frase1
    #stampo la frase con probabilita maggiore secondo il modello di Markov di ordine 0
    print ("Frase con probabilità piu alta calcolata attraverso un modello di Markov di ordine 2:")
    print ("Frase: '", fraseProbMax1, "' Probabilità:", probMax1, ".")
 
    print ("\n")
    print ("\n")

    return probMax0






"""dopo aver individuato e classificato le Entità Nominate (NE) presenti nel testo, estraete:
◦ i 15 nomi propri di persona più frequenti, ordinati per frequenza;"""

def Nomi(tokenPOSTot):

    nomitag = nltk.ne_chunk(tokenPOSTot)
    persone = []
    for nodo in nomitag:
        Nomi = ""
        #se il nodo ha l'attributo "label"
        if hasattr(nodo, "label"):
            if nodo.label() in ["PERSON"]:
                #se il valore dell'attributo corrisponde a una persona li inserisce in una lista apposita
                for pNomi in nodo.leaves():
                    Nomi = Nomi + " " + pNomi[0]
                if (nodo.label() == "PERSON"):
                    persone.append(Nomi)

    personePiuFrequenti = nltk.FreqDist(persone).most_common(15)
    return personePiuFrequenti




def main(file1, file2):
    open1= open(file1, mode="r", encoding="utf-8") #apro il file1 con modalità lettura ed estensione utf-8
    open2= open(file2, mode="r", encoding= "utf-8") #apro il file2 con modalità lettura ed estensione utf-8
    read1= open1.read() #mi apre e legge il file1
    read2= open2.read() #mi apre e legge il file2
    tokenizer= nltk.data.load('tokenizers/punkt/english.pickle')
    test1= tokenizer.tokenize(read1)
    test2= tokenizer.tokenize(read2)
    listaTokens1, listaPoS1, lunghezzaTesto1= annotation(test1)
    listaTokens2, listaPoS2, lunghezzaTesto2= annotation(test2)
    frequency1= FrequenzaPoS(listaPoS1)
    frequency2= FrequenzaPoS(listaPoS2)
    print("-- LE 10 POS PIU' FREQUENTI --")
    print("Le 10 PoS più frequenti in:", file1, "sono:\n", frequency1)
    print()
    print("Le 10 PoS più frequenti in:", file2, "sono:\n", frequency2)
    print()
    Adjective1= FreqParole(listaPoS1)
    Adjective2= FreqParole(listaPoS2)
    print("-- I 20 AGGETTIVI PIU' FREQUENTI --")
    print("I 20 aggettivi più frequenti in", file1, "sono:\n", Adjective1)
    print()
    print("I 20 aggettivi più frequenti in", file2, "sono:\n", Adjective2)
    print()
    Adverbs1= FreqAdv(listaPoS1)
    Adverbs2= FreqAdv(listaPoS2)
    print("-- I 20 AVVERBI PIU' FREQUENTI --")
    print("I 20 avverbi più frequenti in", file1, "sono:\n", Adverbs1)
    print()
    print("I 20 avverbi più frequenti in", file2, "sono:\n", Adverbs2)
    print()
    frequenzaB1= calcoloBigrams(listaPoS1)
    frequenzaB2= calcoloBigrams(listaPoS2)
    print("-- I 10 BIGRAMMI DI POS PIU' FREQUENTI --")
    print("I 10 bigrammi di PoS più frequenti in", file1, "sono:\n", frequenzaB1)
    print()
    print("I 10 bigrammi di PoS più frequenti in", file2, "sono:\n", frequenzaB2)
    print()
    frequenc1= Trigrammi(listaPoS1)
    frequenc2= Trigrammi(listaPoS2)
    print("-- I 10 TRIGRAMMI DI POS PIU' FREQUENTI --")
    print("I 10 trigrammi di PoS più frequenti in", file1, "sono:\n", frequenc1)
    print()
    print("I 10 trigrammi di PoS più frequenti in", file2, "sono:\n", frequenc2)
    print()
    listaSosteAGG(listaPoS1, listaTokens1)
    listaSosteAGG(listaPoS2, listaTokens2)
    frequenzaP1= NamedEntity(listaPoS1)
    frequenzaP2= NamedEntity(listaPoS2)
    print("-- NAMED ENTITY --")
    print("I 15 nomi propri di persona più frequenti in", file1, "sono:\n", frequenzaP1)
    print()
    print("I 15 nomi propri di persona più frequenti in", file2, "sono:\n", frequenzaP2)
    
    
#Faccio una annotazione del testo per ricavarmi poi bigrammi, trigrammi, ...
def annotation(test):
    listaTokens=[] #lista che contiene i tokens
    listaPoS=[] #lista che contiene le POS 
    for frase in test:
        divisionf= nltk.word_tokenize(frase) #divido la frase in token
        PosTagToken= nltk.pos_tag(divisionf) #aggiungo il PoS per svolgere l'analisi morfosintattica
        listaTokens= listaTokens + divisionf #costituita dalla concatenazione tra i token e le frasi divise in tokens
        listaPoS= listaPoS + PosTagToken #lista PoS e tokens
    lunghezzaTesto= len(listaTokens) #numero di tokens totali nel testo
    return listaTokens, listaPoS, lunghezzaTesto

#le 10 PoS più frequenti
def FrequenzaPoS(listaPoS):
    listaSpeech=[] #lista con le POS all'interno, quindi ritroverò le coppie (token,POS)
    for token in listaPoS: #scorro le POS e i miei TOKEN
        listaSpeech.append(token[1]) #appendi, inserisci in posizione 1 la POS inerente quel token
    valuto= nltk.FreqDist(listaSpeech) #utilizzo questa funzione per avere una distribuzione di frequenza inerente le POS
    frequency= valuto.most_common(10) #utilizzo most_common per avere le 10 POS più frequenti
    return frequency

#calcolo i 20 aggettivi e i 20 avverbi più frequenti
def FreqParole(listaPoS):
    Freq=[] #costruisco un array in cui riscontrerò la presenza di [token,POS]
    for token in listaPoS: #scorro tutta la lista delle POS (mi serve per verificare)
        if token[1] in ["JJ", "JJR", "JJS"]: #se quel token equivale a una di queste POS
            Freq.append(token[0]) #aggiungi alla fine della lista il primo token che vedi nella lista 
    DistAdj= nltk.FreqDist(Freq) #calcolo la distribuzione di frequenza delle POS  inerenti gli ADJ
    Adjective= DistAdj.most_common(20) #most_common mi serve per estrapolare le 10 POS riagganciandomi alla frequenza di cui sopra
    return Adjective

def FreqAdv(listaPoS):
    Frequ= [] #costruisco un array vuoto in cui ritroverò le coppie [token,POS]
    for token in listaPoS:
        if token[1] in ["RB", "RBR", "RBS"]: #se quel token è presente e risulta essere queste POS
            Frequ.append(token[0]) #allora aggiungi pure quel token in fondo alla lista 
    DistAdv= nltk.FreqDist(Frequ) #calcolo al distribuziione di frequenza delle POS inerenti gli ADV
    Adverbs= DistAdv.most_common(20) #estrapolo le 20 POS più frequenti
    return Adverbs

#calcolo 10 bigrams e trigrams più frequenti
def calcoloBigrams(listaPoS):
    bigrammi= bigrams(listaPoS) #estraggo bigrammi 
    distribution= nltk.FreqDist(bigrammi) #so che le POS sono costituite da (token,POS)
    frequenzaB= distribution.most_common(10) #richiedo che mi dia come output solo i 10 più frequenti
    return frequenzaB 

#calcolo i trigrams 
def Trigrammi(listaPoS):
    trigrammi= trigrams(listaPoS) #estraggo i trigrammi 
    trigramsDistribution= nltk.FreqDist(trigrammi) #calcolo la distribuzione di frequenza dei trigrammi
    frequenc= trigramsDistribution.most_common(10) #voglio i 10 più frequenti
    return frequenc

#estrarre 20 bigrammi composti da aggettivo-sostantivo (freq >3)
def listaSosteAGG(listaPoS,listaTokens):
    lista=[] #lista che contiene i miei bigrammi, quindi la sequenza: aggettivo, sostantivo
    listaLMI=[] #lista della LMI cosituita da: (bigramma e LMI)
    ListaPCond= [] #lista Probabilità condizionata che sarà costituita da: bigramma, prob. condizionata
    bigrammiTokens= list(bigrams(listaTokens)) #restituisce i bigrammi 
    listab= list(bigrams(listaPoS)) #lista bigrams 
    for token in listab: #vado a scorrere i bigrammi che ho estratto dalle PoS 
        if (token[0][1] in ["JJ", "JJR", "JJS"]) and (token[1][1] in ["NNP", "NNPS", "NN", "NNS"]): #se è aggettivo e sostantivo
            frequenza1= listaTokens.count(token[0][0]) #conto prendendo in considerazione il token in posizione [0][0]
            frequenza2= listaTokens.count(token[1][0]) #conto prendendo in considerazione il token in posizione [1][0]
            if frequenza1 > 3 and frequenza2 > 3: #prendo in considerazione solo frequenze maggiori di 3
                lista.append((token[0][0], token[1][0])) #se le trova, allora chiedo di aggiungere dentro lista
    bigrammii= nltk.FreqDist(lista) #utilizzo la distribuzione di frequenza considerando lista
    bigrams20= bigrammii.most_common(20) #voglio i 20 più frequenti
    bigrammiDiversi= list(set(lista)) #tolgo le ripetizioni
    for token in bigrammiDiversi:
        frequBigram= bigrammiTokens.count(token) #frequenza bigrammi (che calcolo andando a contare)
        frequenzaElemento1= listaTokens.count(token[0]) #frequenza primo elemento bigram
        frequenzaElemento2= listaTokens.count(token[1]) #frequenza secondo elemento bigram
        ProbCondizionata= frequBigram/frequenzaElemento1 #la probabilità condizionata è P(A,B)/P(B)
        ListaPCond.append((token, ProbCondizionata)) #la aggiungo all'interno della lista 
        probabilitàtoken1= frequenzaElemento1/len(listaTokens) #frequenza dell'elemento e corpus
        probabilitàtoken2= frequenzaElemento2/len(listaTokens) #frequenza dell'altro elemento e corpus
        ProbCongiunta= ProbCondizionata*probabilitàtoken1 #faccio riferimento alla regola del prodotto P(A,B)=P(A)*P(B)
        Probabilità= ProbCongiunta/(probabilitàtoken1*probabilitàtoken2) #la calcolo per calcolare la MI
        MI= math.log(Probabilità,2) #gli dico che il logaritmo è in base due e che prenda in considerazione la variabile probabilità
        LMI= frequBigram*MI #sappiamo essere la MI*frequenzaosservata, quindi p(<u,v>)*MI
        listaLMI.append((token, LMI)) #la aggiungo all'interno della lista 
    Ordinamento= sorted(ListaPCond, reverse= True, key= lambda x:x[1]) #ordinamento decrescente sulla base della p. condizionata
    Primi20= Ordinamento[:20] #la soglia che non deve superare è 20 e ottengo i 20 bigrammi (bigramma, P.Cond)
    OrdinamentoLMI= sorted(listaLMI, reverse=True, key= lambda x:x[1]) #ordinamento decrescente per la LMI
    Altri20= OrdinamentoLMI[:20] #ottengo i 20 bigrammi (bigramma, LMI)
    #Stampo i vari risultati, prendendo in considerazione il primo elemento e dando come risultato il secondo [token[0]],[tonen[1]]
    print("-- LMI --")
    print("La LMI massima, è:\n")
    for token in Altri20:
        print("Il bigramma", token[0], "con LMI:", token[1]) #prende primo elemnento (token[0])  e poi secondo elemento (token[1])
        print()
    print("-- PROBABILITA' CONDIZIONATA MASSIMA --")
    print("La Proababilità condizionata maxssima dei bigrammi  è:\n")
    for token in Primi20:
        print("Il bigramma", token[0], "con probabilità:", token[1])
        print()
    print("-- I 20 BIGRAMMI COMPOSTI DA SOST. E AGG. --")
    print("I 20 bigrammi composti da AGG. e SOST. (i cui token hanno frequenza > di 3) più frequenti sono:\n")
    for token in bigrams20:
        print("Il bigramma", token[0], "con frequenza:", token[1])
        print()

#individuiamo le NAMED ENTITY ed estraiamo i 15 nomi propri (PERSON)
def NamedEntity(listaPoS): 
    Named= nltk.ne_chunk(listaPoS) #mi serve questa funzione per identiifcare e classificare delle ENTITA' NOMINATE
    nomipropri=[] #ci sono tre classi, ma in questo caso ci interessa PERSON (conterrà tutte le NAMED ENTITY PERSON)
    for nodo in Named: #scorro l'albero nodo per nodo
        NaEn=""
        if hasattr(nodo, "label"): #controlla se nodo (chunk) è un nodo intermedio oppure foglia
            if nodo.label() in ["PERSON"]: #in questo modo estraggo l'etichetta del nodo. Mi interessa che la NAMED ENTITY sia PERSON
                for partNE in nodo.leaves(): #ciclo le foglie del nodo selezionato e ottengo le liste delle foglie del nodo chunk
                    NaEn= NaEn+" "+partNE[0] 
                if (nodo.label()=="PERSON"): #se allora questo nodo è una PERSON, quindi equivale a questa ENTITA' NOMINATA
                    nomipropri.append(NaEn) #allora chiedo di aggiungere le NAMED ENTITY all'interno della lista nomipropri
    frequenzaP= nltk.FreqDist(nomipropri).most_common(15) #mi interessano i 15 nomi propri, quindi utilizzo la funzione most_common
    return frequenzaP




    

main(sys.argv[1], sys.argv[2])
    


    
    
