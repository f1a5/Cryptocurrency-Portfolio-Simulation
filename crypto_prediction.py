
#
# crypto_prediction.py
#
# full implementation of the cryptocurrency prediction by using numerical optimisation

# imports needed to make this script work

import csv
import math
import random
from simanneal import Annealer
from deap import base
from deap import creator
from deap import tools
import timeit
import matplotlib.pyplot as plt
import copy

class Portfolio:
    def __init__(self,Funds, Closing_Values) -> None:
        self.funds = copy.copy(Funds)
        self.bitcoin_buy_weights = [0.01, 0.01 ,0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
        self.bitcoin_sell_weights = [-0.01, -0.01 ,-0.01, -0.01, -0.01, -0.01, -0.01, -0.01]
        self.dogecoin_buy_weights = [0.01, 0.01 ,0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
        self.dogecoin_sell_weights = [-0.01, -0.01 ,-0.01, -0.01, -0.01, -0.01, -0.01, -0.01]
        self.etherium_buy_weights = [0.01, 0.01 ,0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
        self.etherium_sell_weights = [-0.01, -0.01 ,-0.01, -0.01, -0.01, -0.01, -0.01, -0.01]
        self.xrp_buy_weights = [0.01, 0.01 ,0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
        self.xrp_sell_weights = [-0.01, -0.01 ,-0.01, -0.01, -0.01, -0.01, -0.01, -0.01]
        self.litecoin_buy_weights = [0.01, 0.01 ,0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
        self.litecoin_sell_weights = [-0.01, -0.01 ,-0.01, -0.01, -0.01, -0.01, -0.01, -0.01]
        self.coins = [0, 0, 0, 0, 0]
        self.current_coin_prices = [0, 0, 0, 0, 0]
        self.closing_values = Closing_Values
        self.price_per_coin = [0,0,0,0,0]
        self.funds_per_slot = Funds
        self.daily_value = []
        self.capital_gains =  0
        self.profit_loss = self.preCalculateProfitLoss(self.closing_values)
        self.bitcoin_profit_loss = self.profit_loss[:][0][:]
        self.dogecoin_profit_loss = self.profit_loss[:][1][:]
        self.ethereum_profit_loss = self.profit_loss[:][2][:]
        self.litecoin_profit_loss = self.profit_loss[:][3][:]
        self.xrp_profit_loss = self.profit_loss[:][4][:]
        self.output_file = "output_Genetic7.txt"
        self.coin_names = ["Bitcoin", "Dogecoin", "Etherium", "Litecoin", "XRP"]
        self.capital_gains_tax = 0.15
    # DONE
    def determineBuySignals(self, bitcoin, dogecoin, etherium, litecoin, xrp):
        #create tuples of corresponding weights with the pl and then count total signals
        bitcoin_signals = sum(1 for x in zip(list(filter(lambda x: x is not None, bitcoin)),self.bitcoin_buy_weights) if x[0] > x[1])
        dogecoin_signals = sum(1 for x in zip(list(filter(lambda x: x is not None, dogecoin)),self.dogecoin_buy_weights) if x[0] > x[1])
        etherium_signals = sum(1 for x in zip(list(filter(lambda x: x is not None, etherium)),self.etherium_buy_weights) if x[0] > x[1])
        litecoin_signals = sum(1 for x in zip(list(filter(lambda x: x is not None, litecoin)),self.litecoin_buy_weights) if x[0] > x[1])
        xrp_signals = sum(1 for x in zip(list(filter(lambda x: x is not None, xrp)),self.xrp_buy_weights) if x[0] > x[1])
        
        return bitcoin_signals, dogecoin_signals, etherium_signals,litecoin_signals, xrp_signals

    # DONE
    def determineSellSignals(self, bitcoin, dogecoin, etherium, litecoin, xrp):
        #create tuples of corresponding weights with the pl and then count total signals
        bitcoin_signals = sum(1 for x in zip(list(filter(lambda x: x is not None, bitcoin)),self.bitcoin_sell_weights) if x[0] < x[1])
        dogecoin_signals = sum(1 for x in zip(list(filter(lambda x: x is not None, dogecoin)),self.dogecoin_sell_weights) if x[0] < x[1])
        etherium_signals = sum(1 for x in zip(list(filter(lambda x: x is not None, etherium)),self.etherium_sell_weights) if x[0] < x[1])
        litecoin_signals = sum(1 for x in zip(list(filter(lambda x: x is not None, litecoin)),self.litecoin_sell_weights) if x[0] < x[1])
        xrp_signals = sum(1 for x in zip(list(filter(lambda x: x is not None, xrp)),self.xrp_sell_weights) if x[0] < x[1])
       
        return bitcoin_signals, dogecoin_signals, etherium_signals,litecoin_signals, xrp_signals

    def displayValueGraph(self):
        #Run Simulation Before running this
        print(len(closing_values[1]))
        plt.plot(range(len(self.daily_value)),self.daily_value)
        plt.xlabel("Days")
        plt.ylabel("Total Value of Portfolio $")
        plt.show()
        return
        

    # DONE
    def preCalculateProfitLoss(self, currency_closing):
        precalculatedPL = []
        # first index is day, 2nd is currency, 3rd are weights 0-7
        for j in range(len(currency_closing[1])):
                precalculatedPL.append(self.profitLoss(currency_closing,j+1))
        return precalculatedPL
        
    # DONE
    def profitLoss(self, currency_closing,day):
        PL = []
        for i in range(len(closing_values)):
            temp = []
            for j in range(8):
                if currency_closing[i][day-1-2**j]  == 0 or day-1-2**j<0:
                    temp.append(None)
                else:
                    temp.append((currency_closing[i][day-1] - currency_closing[i][day-1-2**j])/currency_closing[i][day-1-2**j])
            PL.append(temp)
        return PL

    def purchaseCurrency(self, currency, buy_signals, sell_signals, log = True):
        # Currency is 0-4 for bitcoin, dogecoin, etherium, Litecoin and xrp
        if buy_signals[currency] > sell_signals[currency] and self.current_coin_prices[currency]>0.01 and self.coins[currency] == 0:
            # Add the coins according to price and remove the funds corresponding

            self.coins[currency] += self.funds_per_slot[currency]//self.current_coin_prices[currency]
            
            self.funds_per_slot[currency] -= self.coins[currency]*self.current_coin_prices[currency]
            if log == True:
                with open(self.output_file,'a') as f:
                    f.write("Buying currency " + str(self.coin_names[currency]) + " @ " + str(self.current_coin_prices[currency]) +"\n\n")
                
            self.price_per_coin[currency] = self.current_coin_prices[currency]

    def reset(self):
        self.coins = [0, 0, 0, 0, 0]
        self.current_coin_prices = [0, 0, 0, 0, 0]
        self.price_per_coin = [0,0,0,0,0]
        self.funds_per_slot = copy.copy(self.funds)
        self.daily_value = []
        self.capital_gains = 0
        pass

    def sellCurrency(self, currency, buy_signals, sell_signals, log=True):
        # Currency is 0-4 for bitcoin, dogecoin, etherium, Litecoin and xrp
        if sell_signals[currency] >= buy_signals[currency] and self.coins[currency] != 0:
            # Add the amount to funds and set the coins to 0 as all are sold

            self.funds_per_slot[currency] += self.coins[currency]*self.current_coin_prices[currency]

            profit = (self.current_coin_prices[currency] - self.price_per_coin[currency])*self.coins[currency]
            self.capital_gains = self.capital_gains + profit*self.capital_gains_tax if profit>0 else self.capital_gains
            self.coins[currency] = 0;
            if log == True:
                with open(self.output_file,'a') as f:
                    f.write("Selling currency " + str(self.coin_names[currency]) + " @ " + str(self.current_coin_prices[currency]) +"\n")
                    f.write("Profitability :: " + str(profit) +"\n\n")

            self.price_per_coin[currency] = 0;


    def simulate(self,log = True):
        for i in range(128,len(self.closing_values[0])):

            self.current_coin_prices = [row[i] for row in self.closing_values]
            TotalBuySignals  = self.determineBuySignals(self.profit_loss[i][0],self.profit_loss[i][1],self.profit_loss[i][2],self.profit_loss[i][3],self.profit_loss[i][4])
            TotalSellSignals = self.determineSellSignals(self.profit_loss[i][0],self.profit_loss[i][1],self.profit_loss[i][2],self.profit_loss[i][3],self.profit_loss[i][4])
            if log == True:
                with open(self.output_file,'a') as f:
                    f.write("_____Simulating Day " +str(i)+": \n")
                    f.write("Bitcoin \t Dogecoin \t Etherium \t LiteCoin \t XRP \n")
                    f.write('\t'.join(str(TotalBuySignals)) + "\n")
                    f.write('\t'.join(str(TotalSellSignals)) + "\n")
                    f.write("Total Funds : " + str(self.totalValue())+ "\n")

            for currency in range(5):
                self.purchaseCurrency(currency,TotalBuySignals,TotalSellSignals,log)
                self.sellCurrency(currency,TotalBuySignals,TotalSellSignals,log)
            
            self.daily_value.append(self.totalValue())
        
        #sell all currency in the end for value estimation
        for currency in range(5):
            self.sellCurrency(currency,[0,0,0,0,0],[8,8,8,8,8])
        

    def ApplyWeights(self, Weights):
        # Apply Weights easily from 80 float input from annealer or evolution
        self.bitcoin_buy_weights = Weights[0:7]
        self.dogecoin_buy_weights = Weights[8:15]
        self.etherium_buy_weights = Weights[16:23]
        self.litecoin_buy_weights = Weights[24:31]
        self.xrp_buy_weights = Weights[32:39]
        self.bitcoin_sell_weights = Weights[40:47]
        self.dogecoin_sell_weights = Weights[48:55]
        self.etherium_sell_weights = Weights[56:63]
        self.litecoin_sell_weights = Weights[64:71]
        self.xrp_sell_weights = Weights[72:79]
    
    def totalNetValue(self):
        return sum(self.funds_per_slot) - self.capital_gains

    def totalValue(self):
        return sum(self.funds_per_slot)

class CryptoOptimization(Annealer):

    def __init__(self, weights):
        self.weights = weights
        super(CryptoOptimization, self).__init__(weights)
    
    def move(self):
        #Rake random index
        self.state[random.randint(0,79)] = random.uniform(-1,1)

    
    def energy(self):
        
        portfolio = Portfolio([2000,2000,2000,2000, 2000],closing_values)
        portfolio.ApplyWeights(self.state)
        portfolio.simulate(log=False)
        return 1/portfolio.totalValue()                 #so decrease in energy actually improves value



    
# function that will read in all of the cryptocurrenies and will assemble it into workable data
def assembleData(dates, closing_values):
    # read in all of the CSV files to make sure they are working
    bitcoin_rows = readCSVFile('coin_Bitcoin.csv')
    dogecoin_rows = readCSVFile('coin_Dogecoin.csv')
    ethereum_rows = readCSVFile('coin_Ethereum.csv')
    litecoin_rows = readCSVFile('coin_Litecoin.csv')
    xrp_rows = readCSVFile('coin_XRP.csv')

    # create a set of all the dates in all 5 CSVs. skip the first row as it is the header
    # sort the set at the end and turn it into a list
    dateset = set()
    print(dateset)
    for i in range(1, len(bitcoin_rows)):
        dateset.add(stripTime(bitcoin_rows[i][3]))
    for i in range(1, len(dogecoin_rows)):
        dateset.add(stripTime(dogecoin_rows[i][3]))
    for i in range(1, len(ethereum_rows)):
        dateset.add(stripTime(ethereum_rows[i][3]))
    for i in range(1, len(litecoin_rows)):
        dateset.add(stripTime(litecoin_rows[i][3]))
    for i in range(1, len(xrp_rows)):
        dateset.add(stripTime(xrp_rows[i][3]))
    for i in list(sorted(dateset)):
        dates.append(i)

    # create lists for all 5 currencies that contain all of the closing values that are mapped to the correct date
    # the first row will be the date
    for i in range(5):
        closing_values.append([0.0] * len(dateset))

    # add the closing value of all of the currencies into the closing values 2D list
    mapClosingValues(bitcoin_rows, closing_values, dates, 0)
    mapClosingValues(dogecoin_rows, closing_values, dates, 1)
    mapClosingValues(ethereum_rows, closing_values, dates, 2)
    mapClosingValues(litecoin_rows, closing_values, dates, 3)
    mapClosingValues(xrp_rows, closing_values, dates, 4)

# function that will map the closing value data from the given set of rows to the correct dates in the dataset
def mapClosingValues(rows, closing_values, dates, currency):
    # take the first date from the rows as this will give us our starting index.
    # unlike the stock market, crypto markets work on weekends as well so the dates will be in order
    date = stripTime(rows[1][3])
    starting_index = dates.index(date)

    # go through each of the rows and set the closing value. we skip the irst row as it is a header
    for i in range(1, len(rows)):
        closing_values[currency][starting_index + i - 1] = float(rows[i][7])

# function that will take in the given CSV file and will read in its entire contents
# and return a list of lists
def readCSVFile(file):
    # the rows to return
    rows = []

    # open the file for reading and give it to the CSV reader
    csv_file = open(file)
    csv_reader = csv.reader(csv_file, delimiter=',')

    # read in each row and append it to the list of rows.
    for row in csv_reader:
        rows.append(row)

    # close the file when reading is finished
    csv_file.close();

    # return the rows at the end of the function
    return rows

# function that will take a date time and strip out the time from it
def stripTime(str_datetime):
    # datetimes will have the first 10 characters to represent the date so just extract these
    return str_datetime[0:10]

def Anneal(weights, STEPS, Minimum_Temperature ):
    optimize = CryptoOptimization(weights)

    optimize.steps = STEPS
    optimize.Tmin = Minimum_Temperature
    state, e = optimize.anneal()

    return state, e

def evaluate_deap(individual):
    portfolio = Portfolio([2000,2000,2000,2000,2000], closing_values)
    portfolio.ApplyWeights(individual)
    portfolio.simulate(log=False)

    return portfolio.totalValue()

def main_deap(Child_Probability, Mutant_Probability, Number_of_generations):
    pop = toolbox.population(n=30)
    CXPB, MUTPB, NGEN = Child_Probability, Mutant_Probability, Number_of_generations
    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, pop)

    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = [fit]

    for g in range(NGEN):
        print("Current Gen ::"+ str(g))
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = [fit]

        # The population is entirely replaced by the offspring
        pop[:] = offspring

    return pop

def rouletteWheel(a,b):
    # return firs if <0.5, seconf if more than 0.5
    return a if random.random()<0.5 else b

def deapCrossover(parent1, parent2):
    # swaps the elements of inputs depending on rouletteWheel() function
    offspring1 = [];
    offspring2 = [];
    #loop over all elements
    for i in range(len(parent1)):
        offspring1.append(rouletteWheel(parent1[i],parent2[i]))
        # if element from parent 1 is chosen then element from parent 2 goes in 2nd offspring
        # and vice versa
        if offspring1[i] == parent1[i]:
            offspring2.append(parent1[i])
        else:
            offspring2.append(parent1[i])
    # return tuple of both offsprings
    return (offspring1,offspring2)
   

# entry point to the script
if __name__ == '__main__':
    # empty lists for the dates and the closing data we need
    dates = []
    closing_values = []
    
    # assemble the data that we need and run the appropriate optimisation
    assembleData(dates, closing_values)

    portfolio = Portfolio([2000, 2000, 2000, 2000, 2000], closing_values)
    ans1 = portfolio.profit_loss
    

    ans = portfolio.simulate()
    portfolio.displayValueGraph()

    ## Annealing, Boolean to control wether to do or not
    SANNEAL = False
    if SANNEAL == True:
        Initial_weights = [0.01, 0.01, 0.01, 0.01, 0.01,0.01, 0.01, 0.01,
                0.01, 0.01, 0.01, 0.01, 0.01,0.01, 0.01, 0.01,
                0.01, 0.01, 0.01, 0.01, 0.01,0.01, 0.01, 0.01,
                0.01, 0.01, 0.01, 0.01, 0.01,0.01, 0.01, 0.01,
                0.01, 0.01, 0.01, 0.01, 0.01,0.01, 0.01, 0.01,
                -0.01, -0.01, -0.01, -0.01, -0.01,-0.01, -0.01, -0.01,
                -0.01, -0.01, -0.01, -0.01, -0.01,-0.01, -0.01, -0.01,
                -0.01, -0.01, -0.01, -0.01, -0.01,-0.01, -0.01, -0.01,
                -0.01, -0.01, -0.01, -0.01, -0.01,-0.01, -0.01, -0.01,
                -0.01, -0.01, -0.01, -0.01, -0.01,-0.01, -0.01, -0.01,]
        Annealed_weights, e = Anneal(Initial_weights,10000,15000)

        portfolio.reset()
        portfolio.ApplyWeights(Annealed_weights)
        portfolio.output_file = "Annealed_Output.txt"
        portfolio.simulate()
        portfolio.displayValueGraph()
    
    #Genetic Evolution, Boolean to control wether to do or not
    DEAP = True
    if DEAP == True:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        IND_SIZE = 80

        toolbox = base.Toolbox()
        toolbox.register("attribute", random.uniform,-1,1)              #Random number between -1 and 1
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                        toolbox.attribute, n=IND_SIZE)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", deapCrossover)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", evaluate_deap)

        Final_Population = main_deap(1, 1, 30)
        portfolio = Portfolio([2000, 2000, 2000,2000,2000],closing_values)
        Highest_value = 0;
        Best_ind = 0;
        for i in range(len(Final_Population)):
            portfolio.reset()
            portfolio.ApplyWeights(Final_Population[i])
            portfolio.simulate(log = False)
        if portfolio.totalValue() > Highest_value:
            Highest_value = portfolio.totalValue()
            Best_ind = i
        print("Genetic Evolution " + str(i) +" Total Value :: " + str(portfolio.totalValue()) + "\n")

        print("Best Weights ::" + str(Final_Population[Best_ind]))
        portfolio.reset()
        portfolio.ApplyWeights(Final_Population[i])
        portfolio.output_file = "Best_population.txt"
        portfolio.simulate()
        portfolio.displayValueGraph()