import pandas as pd
import numpy as np
import bisect

class LFM:
    def __init__(self, baseTime, T0, initRate, capMarketInfo, tarVol_swaption, opt, handlingOpt, outputfileName):
        self.capMarketInfo = capMarketInfo
        self.baseTime = baseTime
        self.T0 = T0
        self.initRate = initRate
        self.tarVol_swaption = tarVol_swaption
        capMarketInfo['DF'] = self.calDF(baseTime, T0, initRate, capMarketInfo)
        WandSwaprate = self.getSwapAndSwapRates()
        self.W = WandSwaprate['W']
        self.swapRates = WandSwaprate['swapRates']
        self.packetInfo = {'tarVol_swaption':self.tarVol_swaption, 'W': self.W, 'capMarketInfo':capMarketInfo, 'swapRates':self.swapRates, 'opt': opt, 'handling': handlingOpt}
        self.outputfileName = outputfileName
        self.modelPara = {}

    def getModelPara(self):
        return self.modelPara

    def mySave(self, sigma):
        writer = pd.ExcelWriter(self.outputfileName)
        sigma.to_excel(writer)
        writer.save()

    def calDF(self, baseTime, initRateMaturity, initRateValue, capMarketInfo):
        initZcpPrc = (1.0 / (1.0 + (initRateMaturity - baseTime) * initRateValue))
        zcpPrc = (1.0 / (1.0 + capMarketInfo['Tenor'] * capMarketInfo['Fwd']))
        for i in np.arange(1, len(zcpPrc)):
            zcpPrc[zcpPrc.index[i]] = zcpPrc[zcpPrc.index[i - 1]]*zcpPrc[zcpPrc.index[i]]
        return zcpPrc * initZcpPrc

    def getSwapAndSwapRates(self):
        swapRates = pd.DataFrame(index=self.tarVol_swaption.index.values, columns=self.tarVol_swaption.columns.values, data=0.0)
        W = pd.DataFrame(index=self.tarVol_swaption.index.values, columns=self.tarVol_swaption.columns.values)
        for T0 in self.tarVol_swaption.index.values:
            for Tn in self.tarVol_swaption.columns.values:
                resetTn = T0 + Tn
                start = bisect.bisect_left(self.capMarketInfo['Fwd'].index.values, T0)  # start is included
                end = bisect.bisect_left(self.capMarketInfo['Fwd'].index.values, resetTn)  # end is not included
                resets = self.capMarketInfo['Fwd'].index.values[start:end]
                wht = pd.Series(index=resets, data=0.0)
                annuity = np.sum([self.capMarketInfo['Tenor'][reset] * self.capMarketInfo['DF'][reset] for reset in resets])
                for reset in resets:
                    wht[reset] = self.capMarketInfo['Tenor'][reset] * self.capMarketInfo['DF'][reset] / annuity
                swapRates.loc[T0][Tn] = np.sum([wht[reset] * self.capMarketInfo['Fwd'][reset] for reset in resets])
                W.loc[T0][Tn] = wht
        return {'W':W, 'swapRates':swapRates}

    def getCorr(self, corr_para):
        if self.packetInfo['opt'] == 'rank2_theta':
            theta = pd.Series(index=self.capMarketInfo.index.values, data=corr_para)
            return self.getCorr_Rank2(theta)
        elif self.packetInfo['opt'] == 'reb_3':
            return self.getCorr_Reb3(corr_para)

    def getCorr_Reb3(self, corr_para):  # , ccapara): *formula 6.45 in the book, page 250
        [a, b, rho] = corr_para
        capMarketInfo = self.packetInfo['capMarketInfo']
        output = pd.DataFrame(index=self.capMarketInfo.index.values, columns=self.capMarketInfo.index.values, data=0.0)
        for i in np.arange(len(self.capMarketInfo.index)):
            for j in np.arange(len(self.capMarketInfo.index)):
                x = -np.abs(i - j) * (b - a * (np.maximum(i, j) - 1.0))
                output.loc[self.capMarketInfo.index[i], self.capMarketInfo.index[j]] = rho + (1.0 - rho) * np.exp(x)
        return output

    def getCorr_Rank2(self, theta):
        corr = pd.DataFrame(index=theta.index, columns=theta.index)
        for key in theta.keys():
            for col in theta.keys():
                corr.loc[key][col] = np.cos(theta[key] - theta[col])
        return corr

    def calculateLargeSum(self, sigma, FW, corr):
        sigma2 = sigma.dot(sigma.T)
        sigma2Corr = np.multiply(sigma2, corr)
        return (np.array((FW.T).dot(sigma2Corr.dot(FW))))[0][0]

    def CCA(self, corr_para): #, cca_para):
        tarVol_swaption = self.packetInfo['tarVol_swaption']
        capMarketInfo = self.packetInfo['capMarketInfo']
        W = self.packetInfo['W']
        swapRates = self.packetInfo['swapRates']
        corr = self.getCorr(corr_para)
        self.modelPara['rho'] = corr.copy()
        corr_target = corr.loc[tarVol_swaption.index.values, tarVol_swaption.index.values]
        sigma = pd.DataFrame(index = tarVol_swaption.index.values, columns = tarVol_swaption.columns.values, data = 0)
        previousSigma = tarVol_swaption.loc[tarVol_swaption.index.values[0], tarVol_swaption.columns.values[0]]
        sigma.loc[tarVol_swaption.index.values[0], tarVol_swaption.columns.values[0]] = previousSigma
        assert(len(tarVol_swaption.index.values) == len(tarVol_swaption.columns.values) )
        s =len(tarVol_swaption.index.values)
        resetTs = capMarketInfo.index
        F_series = capMarketInfo['Fwd'][resetTs]
        for H in np.arange(1,s):
            for m in np.arange(H+1):
                Tm = resetTs[m]
                Tn = resetTs[H-m]
                w = W.loc[Tm,Tn]
                wCol = w[resetTs[m:H]]
                FCol = F_series[resetTs[m:H]]
                corrCol = corr_target.loc[resetTs[H],resetTs[m:H]]
                sigma_local = sigma.copy()
                sigmaCol = sigma_local.loc[resetTs[m:H], resetTs[m]]
                F = F_series[w.index]
                FW = np.matrix((F*w).values).T
                corr = corr_target.loc[w.index ,w.index]
                corr = np.matrix(corr.values)
                sigma_local = sigma_local.loc[w.index, sigma_local.columns[0:m+1]]
                sigma_local = np.matrix(sigma_local.values)
                K = self.calculateLargeSum(sigma_local, FW, corr)
                A = F_series[resetTs[H]]**2 * w[resetTs[H]]**2
                B = 2* (sigmaCol* corrCol * FCol *wCol).sum() * w[resetTs[H]]*F_series[resetTs[H]]
                C= K - Tm * tarVol_swaption.loc[Tm, Tn] **2 * swapRates.loc[Tm, Tn] **2
                x = (-B+ (B**2 - 4*A*C)** 0.5)/(2*A)
                if self.packetInfo['handling'] == 'True':
                    if np.isnan(x) or x<0:
                        sigma.loc[resetTs[H], resetTs[m]] = previousSigma
                    else:
                        previousSigma = x
                        sigma.loc[resetTs[H], resetTs[m]] = x
                else:
                    sigma.loc[resetTs[H], resetTs[m]] = x
        self.modelPara['sigma'] = sigma.copy()
        return sigma
