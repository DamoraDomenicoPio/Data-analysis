% Funzione per tracciare la decision boundary
function plotDecisionBoundary(D, beta_j, beta0, color)
    % Estrai feature 1 dal dataset
    x1_values = D(:, 1);
    
    % Calcola i corrispondenti valori di x2 sulla decision boundary
    x2_values = -(beta_j(1) * x1_values + beta0) / beta_j(2);
    
    % Traccia la decision boundary con il colore specificato
    plot(x1_values, x2_values, 'LineWidth', 2, 'Color', color);
end

