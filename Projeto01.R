# TRAIN SAMPLE

# Detecção de Fraudes no Tráfego de Cliques em Propagandas de Aplicações Mobile
# Projeto 1 - Formação Cientista de Dados da Data Science Academy


# Definição do Problema de Negócio:

# O risco de fraude está em toda parte, mas para as empresas que anunciam online, a fraude
# de cliques pode acontecer em um volume avassalador, resultando em dados de cliques 
# enganosos e dinheiro desperdiçado. Os canais de anúncios podem aumentar os custos 
# simplesmente quando pessoas ou bots clicam nos anúncios em grande escala, o que na 
# prática não gera o resultado esperado. Com mais de 1 bilhão de dispositivos móveis em 
# uso todos os meses, a China é o maior mercado móvel do mundo e, portanto, sofre com 
# grandes volumes de tráfego fraudulento.
# A TalkingData (https://www.talkingdata.com), a maior plataforma de Big Data 
# independente da China, cobre mais de 70% dos dispositivos móveis ativos em todo o país.
# Eles lidam com 3 bilhões de cliques por dia, dos quais 90% são potencialmente 
# fraudulentos. Sua abordagem atual para impedir fraudes de cliques para desenvolvedores 
# de aplicativos é medir a jornada do clique de um usuário em todo o portfólio e sinalizar
# endereços IP que produzem muitos cliques, mas nunca acabam instalando aplicativos. 
# Com essas informações, eles criaram uma lista negra de IPs e uma lista negra de 
# dispositivos.
# Embora bem-sucedidos, eles querem estar sempre um passo à frente dos fraudadores e 
# pediram a sua ajuda para desenvolver ainda mais a solução. Você está desafiado a criar
# um algoritmo que possa prever se um usuário fará o download de um aplicativo depois de 
# clicar em um anúncio de aplicativo para dispositivos móveis.
# Em resumo, neste projeto, você deverá construir um modelo de aprendizado de máquina 
# para determinar se um clique é fraudulento ou não.

# Link dataset disponível no kaggle:
# https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/data


# Dicionário de variáveis:

#  -ip: ip address of click
#  -app: app id for marketing
#  -device: device type id of user mobile phone (e.g., iphone 6 plus, iphone 7, huawei 
# mate 7, etc.)
#  -os: os version id of user mobile phone
#  -channel: channel id of mobile ad publisher
#  -click_time: timestamp of click (UTC)
#  -attributed_time: if user download the app for after clicking an ad, this is the time 
# of the app download
#  -is_attributed: the target that is to be predicted, indicating the app was downloaded
#  -click_id: reference for making predictions

 

# Preparando ambiente / Carregando Pacotes
memory.limit (9999999999)
library(lubridate)
library(ggplot2)
library(rpart)
library(e1071)

# Carregando os dados
df <- read.csv("train_sample.csv")
View(df)

# Análise prévia dos dados e transformação
str(df)

df$ip <- as.factor(df$ip)
df$app <- as.factor(df$app)
df$device <- as.factor(df$device)
df$os <- as.factor(df$os)
df$channel <- as.factor(df$channel)
df$is_attributed <- as.factor(df$is_attributed)
df$click_time <- ymd_hms(df$click_time)
df$attributed_time <- ymd_hms(df$attributed_time)
str(df)

# Gráficos para análise de cada coluna do dataset

# IP
ggplot(df, aes(ip)) + stat_count(geom = "bar", show.legend = TRUE) + 
    ggtitle("Barplot of IP")
table_ip <- table(df$ip)
barplot(table_ip, main = "Frequencies of IP")

# APP
ggplot(df, aes(app)) + stat_count(geom = "bar", show.legend = TRUE) + 
    ggtitle("Barplot of App")
table_app <- table(df$app)
barplot(table_app, main = "Frequencies of App")

# DEVICE

ggplot(df, aes(device)) + stat_count(geom = "bar", show.legend = TRUE) + 
    ggtitle("Barplot of Device")
table_device <- table(df$device)
barplot(table_device, main = "Frequencies of Device")

# OS

ggplot(df, aes(os)) + stat_count(geom = "bar", show.legend = TRUE) + 
    ggtitle("Barplot of OS")
table_os <- table(df$os)
barplot(table_os, main = "Frequencies of OS")

# CHANNEL

ggplot(df, aes(channel)) + stat_count(geom = "bar", show.legend = TRUE) + 
    ggtitle("Barplot of Channel")
table_channel <- table(df$channel)
barplot(table_channel, main = "Frequencies of Channel")

# CLICK_TIME
summary(hour(df$click_time))
table_hour <- table(hour(df$click_time))
barplot(table_hour, main = "Frequencies of Click Hour")

table_month_day <- table(mday(df$click_time))
barplot(table_month_day, main = "Frequencies of Click Month Day")

table_week_day <- table(wday(df$click_time, label = TRUE))
barplot(table_week_day, main = "Frequencies of Click Week Day")
?wday

# ATTRIBUTED_TIME
summary(df$attributed_time)
table_at_hour <- table(hour(df$attributed_time))
barplot(table_at_hour, main = "Frequencies of Attributed Hour")

table_at_mday <- table(mday(df$attributed_time))
barplot(table_at_mday, main = "Frequencies of Attributed Month Day")

table_at_wday <- table(wday(df$attributed_time, label = TRUE))
barplot(table_at_wday, main = "Frequencies of Attributed Week Day")

# Separando dados de treino e de teste
indexes <- sample(1:nrow(df), size = 0.6 * nrow(df))
train_data <- df[indexes,]
test_data <- df[-indexes,]

# Modelo 1
rpart_1 <- rpart(is_attributed ~ ip + app + device + os + channel + click_time, 
                 data = train_data, control = rpart.control(cp = .0005))
rpart_p <- predict(rpart_1, test_data, type = 'class')
confmatrix_rpart1 <- table(rpart_p, true = test_data$is_attributed)
confmatrix_rpart1
mean_rpart <- mean(rpart_p == test_data$is_attributed)
mean_rpart

# Modelo 2
naiveb_1 <- naiveBayes(is_attributed ~ ip + app + device + os + channel + click_time,
                       data = df)
naiveb_p <- predict(naiveb_1, test_data)
confmatrix_naiveb_1 <- table(naiveb_p, true = test_data$is_attributed)
confmatrix_naiveb_1
mean_naiveb <- mean(naiveb_p == test_data$is_attributed)
mean_naiveb

# Como forma de melhorar meus modelos, acredito que o próximo passo seria fazer 
# bootstraping
table(df$is_attributed)
# Pode-se observar que o número de casos de download são muito menores do que o 
# número de casos contrários, o que pode influenciar o treinamento do modelo.
# Não consegui utilizar o comando boot() da biblioteca boot. Não entendi o que 
# deveria colocar no argumento formula. Esse foi o único método que encontrei ao fazer
# pesquisas na internet. Agradeceria se pudesse me dar um feedback sobre como fazer
# bootstraping no R
