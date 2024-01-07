clear all;
clc;
% Parametri
num_samples = 1000;  % Numero totale di campioni
sigma = 1;         % Varianza comune per entrambe le classi

% Generazione di dati per la classe 1
mu_class1 = [0, 0];  % Media per la classe 1
data_class1 = sigma * randn(num_samples, 2) + mu_class1;

% Generazione di dati per la classe 2
mu_class2 = [2, 2];  % Media per la classe 2
data_class2 = sigma * randn(num_samples, 2) + mu_class2;

% Creazione del vettore di etichette (0 per la classe 1, 1 per la classe 2)
labels = [zeros(num_samples, 1); ones(num_samples, 1)];

D = [[data_class1;data_class2],labels];

% Inizializzazione dei parametri per il learning rate costante
beta_j_constant_lr = randn(1, 2);  % Inizializza pesi per due features (beta1, beta2)
beta0_constant_lr = randn();     % Inizializza intercetta (beta0)
learning_rate_constant_lr = 0.01; % Tasso di apprendimento iniziale

% Definizione della funzione logistica (Con @ si indica una funzione anonima 
% dove tra parentesi ci sono i parametri di tale funzione)
logistic_function = @(X, beta_j, beta0) 1 ./ (1 + exp(-(X * beta_j' + beta0)));

% Definizione della funzione di costo (log-likelihood negativa)
log_likelihood = @(X, y, beta_j, beta0) -sum(y .* log(logistic_function(X, beta_j, beta0)) + (1 - y) .* log(1 - logistic_function(X, beta_j, beta0)));

% Implementazione di SGD per l'addestramento
num_iterations = 10000;

% Inizializzazione del vettore per registrare i costi
costs_constant_lr = zeros(num_iterations, 1);

for iteration = 1:num_iterations
    % Scegli una singola osservazione casualmente
    random_index = randi(size(D, 1));
    X_i = D(random_index, 1:2);
    y_i = D(random_index, 3);
    
    % Calcola la probabilità predetta
    p_i = logistic_function(X_i, beta_j_constant_lr, beta0_constant_lr);
    
    % Calcolo derivata
    gradient_beta_j = -(y_i - p_i) * X_i;
    gradient_beta0 = -(y_i - p_i);
    
    % Aggiorna i parametri utilizzando SGD con learning rate costante
    beta_j_constant_lr = beta_j_constant_lr - learning_rate_constant_lr * gradient_beta_j;
    beta0_constant_lr = beta0_constant_lr - learning_rate_constant_lr * gradient_beta0;
    
    % Calcola il costo ad ogni iterazione e registralo
    cost_constant_lr = log_likelihood(D(:, 1:2), D(:, 3), beta_j_constant_lr, beta0_constant_lr);
    costs_constant_lr(iteration) = cost_constant_lr;
end

% Implementazione di SGD per il decaying step size
% Reset dei parametri
beta_j_decaying_lr = randn(1, 2);
beta0_decaying_lr = randn();
learning_rate_decaying_lr = 0.01;
decay_rate = 0.999999;     % Tasso di decadimento

% Inizializzazione del vettore per registrare i costi
costs_decaying_lr = zeros(num_iterations, 1);

for iteration = 1:num_iterations
    % Scegli una singola osservazione casualmente
    random_index = randi(size(D, 1));
    X_i = D(random_index, 1:2);
    y_i = D(random_index, 3);
    
    % Calcola la probabilità predetta
    p_i = logistic_function(X_i, beta_j_decaying_lr, beta0_decaying_lr);
    
    % Calcolo derivata
    gradient_beta_j = -(y_i - p_i) * X_i;
    gradient_beta0 = -(y_i - p_i);
    
    % Aggiorna i parametri utilizzando SGD con decaying step size
    beta_j_decaying_lr = beta_j_decaying_lr - learning_rate_decaying_lr * gradient_beta_j;
    beta0_decaying_lr = beta0_decaying_lr - learning_rate_decaying_lr * gradient_beta0;
    
    % Calcola il costo ad ogni iterazione e registralo
    cost_decaying_lr = log_likelihood(D(:, 1:2), D(:, 3), beta_j_decaying_lr, beta0_decaying_lr);
    costs_decaying_lr(iteration) = cost_decaying_lr;
    
    % Aggiorna il learning rate con decadimento
    learning_rate_decaying_lr = learning_rate_decaying_lr * decay_rate;
end


% Visualizzazione dei dati generati
scatter(data_class1(:, 1), data_class1(:, 2), 'b', 'filled');
grid on;
hold on;
scatter(data_class2(:, 1), data_class2(:, 2), 'r', 'filled');
xlabel('Feature 1');
ylabel('Feature 2');
legend('Classe 1', 'Classe 2');
title('Dataset Generato con Distribuzione Gaussiana');

% Visualizzazione delle decision boundary per il learning rate costante
plotDecisionBoundary(D, beta_j_constant_lr, beta0_constant_lr,'g');

% Visualizzazione delle decision boundary per il decaying step size
plotDecisionBoundary(D, beta_j_decaying_lr, beta0_decaying_lr, 'k');

% Visualizzazione dell'andamento del costo
figure;
plot(1:num_iterations, costs_constant_lr, 'r-', 'LineWidth', 2, 'DisplayName', 'Learning Rate Costante');
hold on;
plot(1:num_iterations, costs_decaying_lr, 'b-', 'LineWidth', 2, 'DisplayName', 'Decaying Step Size');
xlabel('Iterazione');
ylabel('Costo');
title('Andamento del Costo durante Addestramento');
grid on;
legend('show')

%% Valutazione del modello
test_samples = 10;  % Numero totale di campioni nel test set
sigma_test = 1;    % Varianza per il test set

% Generazione del test set
data_test = [sigma_test * randn(test_samples, 2) + mu_class1; sigma_test * randn(test_samples, 2) + mu_class2];
labels_test = [zeros(test_samples, 1); ones(test_samples, 1)];
test_set = [data_test, labels_test];

% Valutazione del modello con learning rate costante
predictions_constant_lr = logistic_function(data_test, beta_j_constant_lr, beta0_constant_lr) >= 0.5;
[accuracy_constant_lr, TP_constant_lr, FP_constant_lr, TN_constant_lr, FN_constant_lr] = evaluateModel(predictions_constant_lr, labels_test);
fprintf('Accuracy (Learning Rate Costante): %.4f\n', accuracy_constant_lr);

% Visualizzazione dei risultati per il modello con learning rate costante
figure;
scatterResults(data_test, predictions_constant_lr, labels_test);
title('Risultati della Classificazione (Learning Rate Costante)');
% Visualizzazione delle decision boundary per il learning rate costante
plotDecisionBoundary(D, beta_j_constant_lr, beta0_constant_lr,'k');


% Valutazione del modello con learning rate decrescente
predictions_decaying_lr = logistic_function(data_test, beta_j_decaying_lr, beta0_decaying_lr) >= 0.5;
[accuracy_decaying_lr, TP_decaying_lr, FP_decaying_lr, TN_decaying_lr, FN_decaying_lr] = evaluateModel(predictions_decaying_lr, labels_test);
fprintf('Accuracy (Learning Rate Decrescente): %.4f\n', accuracy_decaying_lr);

% Visualizzazione dei risultati per il modello con learning rate decrescente
figure;
scatterResults(data_test, predictions_decaying_lr, labels_test);
title('Risultati della Classificazione (Learning Rate Decrescente)');
% Visualizzazione delle decision boundary per il decaying step size
plotDecisionBoundary(D, beta_j_decaying_lr, beta0_decaying_lr, 'k');

% Funzione per valutare il modello e restituire le metriche
function [accuracy, TP, FP, TN, FN] = evaluateModel(predictions, labels)
    TP = sum(predictions == 1 & labels == 1);
    FP = sum(predictions == 1 & labels == 0);
    TN = sum(predictions == 0 & labels == 0);
    FN = sum(predictions == 0 & labels == 1);
    accuracy = (TP + TN) / (TP + FP + TN + FN);
end

% Funzione per visualizzare i risultati della classificazione
function scatterResults(data, predictions, labels)
    scatter(data(predictions == 1 & labels == 1, 1), data(predictions == 1 & labels == 1, 2), 'g', 'filled');
    hold on;
    scatter(data(predictions == 1 & labels == 0, 1), data(predictions == 1 & labels == 0, 2), 'r', 'filled');
    scatter(data(predictions == 0 & labels == 0, 1), data(predictions == 0 & labels == 0, 2), 'b', 'filled');
    scatter(data(predictions == 0 & labels == 1, 1), data(predictions == 0 & labels == 1, 2), 'y', 'filled');
    xlabel('Feature 1');
    ylabel('Feature 2');
    legend('TP', 'FP', 'TN', 'FN');
    title('Risultati della Classificazione sul Test Set');
    grid on;
end


