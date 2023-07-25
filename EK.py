import numpy as np
# Popis rozsireneho Kalmanova filtru


class EKModel:
    """
    Model popisující Lotka-Volterra model
    """
    def __init__(self):
        pass
    def set_D(self, D):
        """
        Nastavi delku casoveho kroku D.
        """
        self.D = D
    def fk(self, k, x):
        pass
    def Fk(self, k, x):
        pass
    def gk(self, k, x):
        pass
    def Gk(self, k):
        pass
    def Qk(self, k):
        pass
    def Rk(self, k, x):
        pass

class EKSystem:
    """
        System pro simulaci v EK filtru
    """
    def __init__(self, model, x0):
        """
        model (instance tridy EKModel): model systemu
        x0 (ndarray): pocatecni stav 
        """

        self.model = model  
        self.x0 = x0        
        self.reset()
 
    def reset(self):
        """
        pocatecni nastaveni pred simulaci systemu
        """

        self.k = 0           # cas
        self.x = self.x0     # stav v case k
        self.y = None        # pozorovani v case k
        self.tracex = None   # trajektorie stavu (x_0, ..., x_k)
        self.tracey = None   # trajektorie pozorovani (y_0, ..., y_k) 
        self.tracet = None   # trajektorie (spojitetho) casu (0*D, 1*D, ..., k*D) 
    
    def step(self):
        """
        krok simulace
        """

        self.k = self.k+1

        Q = self.model.Qk(self.k)
        fk = self.model.fk(self.k, self.x)
        W = np.random.multivariate_normal(np.zeros(Q.shape[0]), Q)
        self.x = fk+W

        R = self.model.Rk(self.k, self.x)
        V = np.random.multivariate_normal(np.zeros(R.shape[0]), R)
        gk = self.model.gk(self.k, self.x)
        self.y = gk+V

        if self.k == 1:
            self.tracex = self.x[None, :]
            self.tracey = self.y[None, :]
            self.tracet = np.array([self.model.D])
        else:
            self.tracex = np.vstack([self.tracex, self.x])
            self.tracey = np.vstack([self.tracey, self.y])
            self.tracet = np.hstack([self.tracet, self.k*self.model.D])

    def run(self, n):
        """
        provede n kroku simulace systemu pocinaje nasledujicim casem (k+1)
        """

        for i in range(n):
            self.step()

        

class EKFiltr:    
    """
    EK filtr
    """

    def __init__(self, system, mu0, sigma0):
        """
        system (instance tridy EKSystem): system s ulozenou trajektorii pozorovani
        m0 (ndarray): str. hodnota pocatecniho odhadu
        sigma0 (ndarray): kovariancni matice pocatecniho odhadu
        """

        self.system = system        # system s ulozenymi trajektoriemi pozorovani a casu,
                                    # slouzi jako zdroj dat (y_1, ..., y_n)
                                    # instance tridy System
        self.model = system.model   # model systemu pro vypocet filtrace
                                    # instance tridy Model
        self.mu = mu0               # aktualni str. hodnota filtrace
        self.sigma = sigma0         # aktualni kovariancni matice filtrace
        self.k = 0                  # aktualni (diskretni) cas
        self.trace_mu = None        # posloupnost str. hodnot filtraci
        self.trace_mu_p = None      # posloupnost str. hodnot jednokrokovych predikci 
        self.trace_sigma = None     # posloupnost kovariancnich matic filtraci
        self.trace_sigma_p = None   # posloupnost kovariancnich matic filtraci
        
    def step(self):
        """
        krok Kalmanova filtru
        """

        self.k = self.k+1

        f = self.model.fk(self.k, self.mu)
        F = self.model.Fk(self.k, self.mu)
        Q = self.model.Qk(self.k)
        G = self.model.Gk(self.k)
        y = self.system.tracey[int(self.k)-1]
        R = self.model.Rk(self.k, y)
        
        # parametry jednokrokove predikce 
        self.mu_p = f
        self.sigma_p = F.dot(self.sigma).dot(F.T)+Q
        # matice zisku
        self.K = self.sigma_p.dot(G.T).dot(np.linalg.inv(G.dot(self.sigma_p).dot(G.T)+R))
        # aktualizace parametru filtrace
        g = self.model.gk(self.k, self.mu_p)
        self.mu = self.mu_p+self.K.dot(y-g)
        self.sigma = self.sigma_p-self.K.dot(G).dot(self.sigma_p)

        # ulozeni parametru predikce a filtrace
        if self.k == 1:
            self.trace_mu = self.mu[None, :]
            self.trace_mu_p = self.mu_p[None, :]
            self.trace_sigma = self.sigma[None, :, :]
            self.trace_sigma_p = self.sigma_p[None, :, :]
        else:
            self.trace_mu = np.vstack([self.trace_mu, self.mu])
            self.trace_mu_p = np.vstack([self.trace_mu_p, self.mu_p])
            self.trace_sigma = np.vstack([self.trace_sigma, self.sigma[None, :,:]])
            self.trace_sigma_p = np.vstack([self.trace_sigma_p, self.sigma_p[None, :, :]])

    def run(self):
        """
        vypocet filtraci pro celou posloupnost pozorovani 
        """

        for i in self.system.tracet:
            self.step()


