%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 理论正确的2D Roesser模型策略迭代算法
% 解决数据驱动与模型驱动根本性差异问题
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% ================ 函数定义区域 ================

function [K, P, iter, convergence_history] = standard_model_based_PI(A1, A2, B1, B2, pi, tau, Q, R, epsilon, max_iter)
    % 标准模型驱动策略迭代
    
    nx = size(A1, 1);
    
    % 初始化为单位矩阵
    P1 = eye(nx);
    P2 = eye(nx);
    
    % 记录收敛历史
    convergence_history = [];
    K_history = [];
    
    fprintf('模型驱动策略迭代开始...\n');
    
    for iter = 1:max_iter
        P1_old = P1;
        P2_old = P2;
        
        % === 策略改进 ===
        % 计算期望P矩阵（控制器视角）
        EP1_ctrl = tau(1,1)*P1 + tau(1,2)*P2;
        EP2_ctrl = tau(2,1)*P1 + tau(2,2)*P2;
        
        % 最优控制增益
        K1 = (R + B1'*EP1_ctrl*B1) \ (B1'*EP1_ctrl*A1);
        K2 = (R + B2'*EP2_ctrl*B2) \ (B2'*EP2_ctrl*A2);
        
        % 记录K值历史
        K_history = [K_history; [K1, K2]];
        
        % === 策略评估 ===
        % 形成闭环矩阵
        A1_cl = A1 - B1*K1;
        A2_cl = A2 - B2*K2;
        
        % 内层迭代求解耦合黎卡提方程
        P1_new = P1; P2_new = P2;
        for inner_iter = 1:100
            P1_inner_old = P1_new;
            P2_inner_old = P2_new;
            
            % 计算期望P矩阵（系统视角）
            EP1_sys = pi(1,1)*P1_new + pi(1,2)*P2_new;
            EP2_sys = pi(2,1)*P1_new + pi(2,2)*P2_new;
            
            % 更新P矩阵
            P1_new = Q + K1'*R*K1 + A1_cl'*EP1_sys*A1_cl;
            P2_new = Q + K2'*R*K2 + A2_cl'*EP2_sys*A2_cl;
            
            % 内层收敛检查
            if norm(P1_new - P1_inner_old, 'fro') + norm(P2_new - P2_inner_old, 'fro') < 1e-12
                break;
            end
        end
        
        P1 = P1_new;
        P2 = P2_new;
        
        % 外层收敛检查
        change = norm(P1 - P1_old, 'fro') + norm(P2 - P2_old, 'fro');
        convergence_history = [convergence_history; change];
        
        if change < epsilon
            fprintf('模型驱动在第%d次迭代收敛，变化量=%.2e\n', iter, change);
            break;
        end
        
        if mod(iter, 10) == 0
            fprintf('  模型驱动迭代%d: 变化=%.2e\n', iter, change);
        end
    end
    
    K = {K1, K2};
    P = {P1, P2};
    
    % 将历史数据存储到全局变量
    assignin('base', 'K_history_model', K_history);
    assignin('base', 'convergence_history_model', convergence_history);
end

function [K, iter] = theory_matched_data_driven_PI(A1, A2, B1, B2, pi, tau, Q, R, max_iter, enforce_iter_count)
    % 理论匹配的数据驱动策略迭代
    % 当 enforce_iter_count 为 true 时，即便策略提前收敛也会坚持执行到给定的 max_iter，
    % 以便与模型驱动方法保持相同的迭代次数。
    
    if nargin < 10
        enforce_iter_count = false;
    end
    
    nx = size(A1, 1);
    
    % 初始策略：由于系统开环稳定，从0开始最安全，避免了初始K不稳定导致的数据发散
    % K1 = zeros(1, nx); 
    % K2 = zeros(1, nx);
    
    K_history = [];
    % 初始化策略（使用更合理的初始值）
    K1 = [2, 0];  % 接近最优解的初始值
    K2 = [0, 2];
    
    policy_tol = 1e-6;
    
    % 阻尼系数：防止K跳变
    damping = 0.5; 
    
    for iter = 1:max_iter
        K1_old = K1; K2_old = K2;
        
        % === 核心：使用独立采样LSTD求解 P ===
        [P1, P2] = lstd_independent_sampling(A1, A2, B1, B2, pi, tau, Q, R, K1, K2);
        
        % === 策略更新 (利用模型信息更新K) ===
        EP1_ctrl = tau(1,1)*P1 + tau(1,2)*P2;
        EP2_ctrl = tau(2,1)*P1 + tau(2,2)*P2;
        
        K1_calc = (R + B1'*EP1_ctrl*B1) \ (B1'*EP1_ctrl*A1);
        K2_calc = (R + B2'*EP2_ctrl*B2) \ (B2'*EP2_ctrl*A2);
        
        % 加入阻尼，平滑收敛轨迹
        K1 = damping * K1_old + (1-damping) * K1_calc;
        K2 = damping * K2_old + (1-damping) * K2_calc;
        
        % 记录与收敛检查
        K_history = [K_history; [K1, K2]];
        diff = norm(K1 - K1_old) + norm(K2 - K2_old);
        
        fprintf('  Iter %d: 策略变化 = %.2e\n', iter, diff);
        if diff < policy_tol && ~enforce_iter_count
            fprintf('数据驱动在第 %d 次迭代收敛。\n', iter);
            break;
        end
        
        if enforce_iter_count && diff < policy_tol && iter == max_iter
            fprintf('数据驱动在第 %d 次迭代收敛（已对齐迭代次数要求）。\n', iter);
        end
    end
    K = {K1, K2};
    assignin('base', 'K_history_data', K_history);
end

function [P1, P2] = lstd_independent_sampling(A1, A2, B1, B2, pi, tau, Q, R, K1, K2)
     % === 独立状态采样 (Independent State Sampling) ===
    % 不再使用轨迹(Trajectory)，而是每次随机生成全新的x
    % 这保证了数据的纯净性和矩阵的良态
    
    n_samples = 3000; % 样本量适中即可
    nx = size(A1, 1);
    n_features = nx*(nx+1)/2;
    n_params = n_features * 2;
    
    % 预分配内存
    Phi_stack = zeros(n_samples, n_params);
    Cost_stack = zeros(n_samples, 1);
    
    A_sys = {A1, A2}; B_sys = {B1, B2};
    
    % 生成完全随机的状态，覆盖[-2, 2]空间
    % 比轨迹采样更能捕捉全局特性
    X_random = 2 * randn(nx, n_samples); 
    
    for i = 1:n_samples
        x = X_random(:, i);
        
        sys_mode = randi(2);
        ctrl_mode = randi(2);
        
        % 1. 施加微小的探测噪声 (保证满秩即可，不需要太大)
        if ctrl_mode==1, u_nom = -K1*x; else, u_nom = -K2*x; end
        u_noise = u_nom + 0.05 * randn; % 噪声从 1.0 降为 0.05
        
        % 2. 状态演化
        x_next = A_sys{sys_mode}*x + B_sys{sys_mode}*u_noise;
        
        % 3. 下一时刻模态模拟
        if rand < pi(sys_mode, 1), next_sys = 1; else, next_sys = 2; end
        % 注意：LSTD中下一时刻控制器模态不影响V(x')的特征提取，因为V只关于x
        
        % 4. 提取特征
        phi_t = extract_quadratic_features(x, nx);
        phi_next = extract_quadratic_features(x_next, nx);
        
        % 5. 构建方程 V(x) - V(x') = Cost
        cost_val = x'*Q*x + u_noise'*R*u_noise;
        
        row = zeros(1, n_params);
        % 当前 V(x)
        idx_start = (sys_mode-1)*n_features + 1;
        row(idx_start : idx_start+n_features-1) = phi_t';
        % 下一时刻 -V(x')
        idx_next_start = (next_sys-1)*n_features + 1;
        row(idx_next_start : idx_next_start+n_features-1) = ...
            row(idx_next_start : idx_next_start+n_features-1) - phi_next';
        
        Phi_stack(i, :) = row;
        Cost_stack(i) = cost_val;
    end
    
    % === 正则化最小二乘 ===
    % 使用 lsqminnorm 或 pinv 求解，或者带正则的左除
    lambda = 1e-6; 
    XtX = Phi_stack' * Phi_stack;
    XtY = Phi_stack' * Cost_stack;
    
    % 求解参数
    theta = (XtX + lambda * eye(size(XtX))) \ XtY;
    
    % 重构并强制正定
    theta1 = theta(1:n_features);
    theta2 = theta(n_features+1:end);
    P1 = reconstruct_matrix(theta1, nx);
    P2 = reconstruct_matrix(theta2, nx);
    
    P1 = (P1+P1')/2; 
    P2 = (P2+P2')/2; 
end

function P = reconstruct_matrix(theta, nx)
    P = zeros(nx, nx); idx = 1;
    for i = 1:nx, for j = i:nx
        P(i,j) = theta(idx); P(j,i) = theta(idx); idx = idx + 1;
    end, end
end

function phi = extract_quadratic_features(x, nx)
    % 提取二次特征 phi，使得 V(x) = phi' * theta = x' * P * x
    
    phi = zeros(nx*(nx+1)/2, 1);
    idx = 1;
    
    for i = 1:nx
        for j = i:nx
            if i == j
                phi(idx) = x(i)^2;
            else
                phi(idx) = 2 * x(i) * x(j);  % 系数2是因为P矩阵对称
            end
            idx = idx + 1;
        end
    end
end

function P = reconstruct_matrix_from_params(theta, nx)
    % 从参数向量重构对称矩阵
    
    P = zeros(nx, nx);
    idx = 1;
    
    for i = 1:nx
        for j = i:nx
            P(i, j) = theta(idx);
            P(j, i) = theta(idx);
            idx = idx + 1;
        end
    end
end

function [K, P] = direct_solve_riccati(A1, A2, B1, B2, pi, tau, Q, R)
    % 直接数值求解耦合黎卡提方程（作为ground truth）
    
    nx = size(A1, 1);
    
    % 使用迭代方法求解
    P1 = eye(nx);
    P2 = eye(nx);
    
    max_iter = 1000;
    tol = 1e-15;
    
    fprintf('直接求解黎卡提方程...\n');
    
    for iter = 1:max_iter
        P1_old = P1;
        P2_old = P2;
        
        % 计算最优增益
        EP1_ctrl = tau(1,1)*P1 + tau(1,2)*P2;
        EP2_ctrl = tau(2,1)*P1 + tau(2,2)*P2;
        
        K1 = (R + B1'*EP1_ctrl*B1) \ (B1'*EP1_ctrl*A1);
        K2 = (R + B2'*EP2_ctrl*B2) \ (B2'*EP2_ctrl*A2);
        
        % 形成闭环矩阵
        A1_cl = A1 - B1*K1;
        A2_cl = A2 - B2*K2;
        
        % 更新P矩阵
        EP1_sys = pi(1,1)*P1 + pi(1,2)*P2;
        EP2_sys = pi(2,1)*P1 + pi(2,2)*P2;
        
        P1 = Q + K1'*R*K1 + A1_cl'*EP1_sys*A1_cl;
        P2 = Q + K2'*R*K2 + A2_cl'*EP2_sys*A2_cl;
        
        % 收敛检查
        change = norm(P1 - P1_old, 'fro') + norm(P2 - P2_old, 'fro');
        if change < tol
            fprintf('直接求解在第%d次迭代收敛，变化量=%.2e\n', iter, change);
            break;
        end
    end
    
    K = {K1, K2};
    P = {P1, P2};
end

function analyze_convergence_theory(A1, A2, B1, B2, pi, tau, Q, R, K, P)
    % 分析收敛性理论
    
    fprintf('收敛性理论分析:\n');
    
    K1 = K{1}; K2 = K{2};
    P1 = P{1}; P2 = P{2};
    
    % 检查最优性条件
    EP1_ctrl = tau(1,1)*P1 + tau(1,2)*P2;
    EP2_ctrl = tau(2,1)*P1 + tau(2,2)*P2;
    
    % 计算梯度（应该为零）
    grad1 = 2*R*K1 + 2*B1'*EP1_ctrl*(A1 - B1*K1);
    grad2 = 2*R*K2 + 2*B2'*EP2_ctrl*(A2 - B2*K2);
    
    fprintf('  最优性条件检查:\n');
    fprintf('    模态1梯度范数: %.2e\n', norm(grad1));
    fprintf('    模态2梯度范数: %.2e\n', norm(grad2));
    
    % 检查黎卡提方程
    A1_cl = A1 - B1*K1;
    A2_cl = A2 - B2*K2;
    
    EP1_sys = pi(1,1)*P1 + pi(1,2)*P2;
    EP2_sys = pi(2,1)*P1 + pi(2,2)*P2;
    
    riccati_res1 = P1 - (Q + K1'*R*K1 + A1_cl'*EP1_sys*A1_cl);
    riccati_res2 = P2 - (Q + K2'*R*K2 + A2_cl'*EP2_sys*A2_cl);
    
    fprintf('  黎卡提方程残差:\n');
    fprintf('    模态1残差范数: %.2e\n', norm(riccati_res1, 'fro'));
    fprintf('    模态2残差范数: %.2e\n', norm(riccati_res2, 'fro'));
    
    % P矩阵条件数
    fprintf('  P矩阵条件数:\n');
    fprintf('    P1条件数: %.2e\n', cond(P1));
    fprintf('    P2条件数: %.2e\n', cond(P2));
end

function P = ensure_positive_definite(P)
    % 确保矩阵正定
    [V, D] = eig(P);
    D = diag(max(diag(D), 0.001));
    P = V * D * V';
    P = (P + P') / 2;  % 确保对称性
end

function verify_stability(A1, A2, B1, B2, K, method_name)
    % 验证闭环系统稳定性
    A1_cl = A1 - B1*K{1};
    A2_cl = A2 - B2*K{2};
    
    eig1 = eig(A1_cl);
    eig2 = eig(A2_cl);
    
    max_eig1 = max(abs(eig1));
    max_eig2 = max(abs(eig2));
    
    fprintf('%s: 模态1最大特征值模=%.8f, 模态2最大特征值模=%.8f', ...
            method_name, max_eig1, max_eig2);
    
    if max_eig1 < 1 && max_eig2 < 1
        fprintf(' [稳定]\n');
    else
        fprintf(' [不稳定]\n');
    end
end

function x_grid = simulate_closed_loop(A, B, K, pi, tau, I_max, J_max)
    % 闭环系统仿真
    nx = size(A{1}, 1);
    x_grid = zeros(nx, I_max, J_max);
    
    % 固定边界条件
    for j = 1:J_max
        if j <= 10
            x_grid(1, 1, j) = 0.8;
        end
    end
    for i = 1:I_max
        if i <= 10
            x_grid(2, i, 1) = 0.7;
        end
    end
    
    % 模态初始化
    sys_mode = ones(I_max, J_max);
    ctrl_mode = ones(I_max, J_max);
    
    % 固定随机种子
    rng(42);
    
    % 仿真循环
    for i = 1:I_max-1
        for j = 1:J_max-1
            x_current = x_grid(:, i, j);
            
            current_sys_mode = sys_mode(i, j);
            current_ctrl_mode = ctrl_mode(i, j);
            
            % 模态转移
            if rand < pi(current_sys_mode, 1)
                next_sys_mode = 1;
            else
                next_sys_mode = 2;
            end
            
            if rand < tau(current_ctrl_mode, 1)
                next_ctrl_mode = 1;
            else
                next_ctrl_mode = 2;
            end
            
            % 更新模态
            if i+1 <= I_max && j <= J_max
                sys_mode(i+1, j) = next_sys_mode;
                ctrl_mode(i+1, j) = next_ctrl_mode;
            end
            if i <= I_max && j+1 <= J_max
                sys_mode(i, j+1) = next_sys_mode;
                ctrl_mode(i, j+1) = next_ctrl_mode;
            end
            
            % 控制输入
            if current_ctrl_mode == 1
                u = -K{1} * x_current;
            else
                u = -K{2} * x_current;
            end
            
            % 状态更新
            x_next = A{current_sys_mode} * x_current + B{current_sys_mode} * u;
            
            if i+1 <= I_max
                x_grid(1, i+1, j) = x_next(1);
            end
            if j+1 <= J_max
                x_grid(2, i, j+1) = x_next(2);
            end
        end
    end
end

function plot_comprehensive_results(x_grid_model, x_grid_data, K_model, K_data, iter_model, iter_data)
    % 绘制单独的比较结果图
    
    % 获取历史数据
    try
        K_history_model = evalin('base', 'K_history_model');
        convergence_history_model = evalin('base', 'convergence_history_model');
        K_history_data = evalin('base', 'K_history_data');
        policy_change_history_data = evalin('base', 'policy_change_history_data');
    catch
        fprintf('警告：无法获取收敛历史数据\n');
        K_history_model = [];
        K_history_data = [];
        convergence_history_model = [];
        policy_change_history_data = [];
    end
    
    % 提取状态数据
    x1_model = squeeze(x_grid_model(1, :, :))';
    x2_model = squeeze(x_grid_model(2, :, :))';
    x1_data = squeeze(x_grid_data(1, :, :))';
    x2_data = squeeze(x_grid_data(2, :, :))';
    
    [I, J] = size(x1_model);
    [m, n] = meshgrid(1:J, 1:I);
    
    %% 图1：模型驱动 - 水平状态 x^h
    figure('Position', [100, 100, 800, 600], 'Name', '模型驱动 - 水平状态演化');
    mesh(m, n, x1_model, 'FaceAlpha', 0.8, 'LineWidth', 0.8);

    xlabel('j方向', 'FontSize', 14); 
    ylabel('i方向', 'FontSize', 14); 
    zlabel('x^h', 'FontSize', 14);
    colorbar; 
    grid on;
    % view(45, 30);
    set(gca, 'FontSize', 12);
    
    %% 图2：模型驱动 - 垂直状态 x^v
    figure('Position', [100, 100, 800, 600], 'Name', '模型驱动 - 垂直状态演化');
    mesh(m, n, x2_model, 'FaceAlpha', 0.8, 'LineWidth', 0.8);

    xlabel('j方向', 'FontSize', 14); 
    ylabel('i方向', 'FontSize', 14); 
    zlabel('x^v', 'FontSize', 14);
    colorbar; 
    grid on;
    % view(45, 30);
    set(gca, 'FontSize', 12);
    
    %% 图3：数据驱动 - 水平状态 x^h
    figure('Position', [100, 100, 800, 600], 'Name', '数据驱动 - 水平状态演化');
    subplot(1,2,1);
    mesh(m, n, x1_data, 'FaceAlpha', 0.8, 'LineWidth', 0.8);
    title('数据驱动方法: x^h(i,j)', 'FontSize', 16, 'FontWeight', 'bold');
    xlabel('j方向', 'FontSize', 14); 
    ylabel('i方向', 'FontSize', 14); 
    zlabel('x^h', 'FontSize', 14);
    colorbar; 
    grid on;
    % view(45, 30);
    set(gca, 'FontSize', 12);
    
    %% 图4：数据驱动 - 垂直状态 x^v
    % figure('Position', [250, 250, 800, 600], 'Name', '数据驱动 - 垂直状态演化');
    subplot(1,2,2);
    mesh(m, n, x2_data, 'FaceAlpha', 0.8, 'LineWidth', 0.8);
    title('数据驱动方法: x^v(i,j)', 'FontSize', 16, 'FontWeight', 'bold');
    xlabel('j方向', 'FontSize', 14); 
    ylabel('i方向', 'FontSize', 14); 
    zlabel('x^v', 'FontSize', 14);
    colorbar; 
    grid on;
    % view(45, 30);
    set(gca, 'FontSize', 12);
    
    %% K1和K2收敛图（差值绝对值）
    % if ~isempty(K_history_model) && !isempty(K_history_data)
    % 
    %     % 计算模型驱动的K值差值绝对值
    %     K_diff_model = zeros(size(K_history_model));
    %     K_diff_model(1, :) = K_history_model(1, :);  % 第一个值保持原值
    %     for i = 2:size(K_history_model, 1)
    %         K_diff_model(i, :) = abs(K_history_model(i, :) - K_history_model(i-1, :));
    %     end
    % 
    %     % 计算数据驱动的K值差值绝对值
    %     K_diff_data = zeros(size(K_history_data));
    %     K_diff_data(1, :) = K_history_data(1, :);  % 第一个值保持原值
    %     for i = 2:size(K_history_data, 1)
    %         K_diff_data(i, :) = abs(K_history_data(i, :) - K_history_data(i-1, :));
    %     end
    % 
    %     iter_model_vec = 0:size(K_history_model, 1)-1;  % 从0开始
    %     iter_data_vec = 0:size(K_history_data, 1)-1;    % 从0开始
    % 
    %     %% 图5：模型驱动 K1收敛过程（差值绝对值）
    %     figure('Position', [300, 300, 800, 600], 'Name', '模型驱动 K1收敛过程');
    %     plot(iter_model_vec, K_diff_model(:, 1), 'r-', 'LineWidth', 2.5, 'Marker', 'o', 'MarkerSize', 6); hold on;
    %     plot(iter_model_vec, K_diff_model(:, 2), 'b--', 'LineWidth', 2.5, 'Marker', 's', 'MarkerSize', 6);
    %     title('模型驱动 K1 收敛过程（初始值+差值绝对值）', 'FontSize', 16, 'FontWeight', 'bold');
    %     xlabel('策略迭代轮数', 'FontSize', 14); 
    %     ylabel('增益值/差值绝对值', 'FontSize', 14);
    %     legend('K1(1) - 水平分量', 'K1(2) - 垂直分量', 'FontSize', 12, 'Location', 'best');
    %     grid on;
    %     set(gca, 'FontSize', 12);
    %     xlim([0, length(iter_model_vec)-1]);
    % 
    %     %% 图6：模型驱动 K2收敛过程（差值绝对值）
    %     figure('Position', [350, 350, 800, 600], 'Name', '模型驱动 K2收敛过程');
    %     plot(iter_model_vec, K_diff_model(:, 3), 'r-', 'LineWidth', 2.5, 'Marker', 'o', 'MarkerSize', 6); hold on;
    %     plot(iter_model_vec, K_diff_model(:, 4), 'b--', 'LineWidth', 2.5, 'Marker', 's', 'MarkerSize', 6);
    %     title('模型驱动 K2 收敛过程（初始值+差值绝对值）', 'FontSize', 16, 'FontWeight', 'bold');
    %     xlabel('策略迭代轮数', 'FontSize', 14); 
    %     ylabel('增益值/差值绝对值', 'FontSize', 14);
    %     legend('K2(1) - 水平分量', 'K2(2) - 垂直分量', 'FontSize', 12, 'Location', 'best');
    %     grid on;
    %     set(gca, 'FontSize', 12);
    %     xlim([0, length(iter_model_vec)-1]);
    % 
    %     %% 图7：数据驱动 K1收敛过程（差值绝对值）
    %     figure('Position', [400, 400, 800, 600], 'Name', '数据驱动 K1收敛过程');
    %     plot(iter_data_vec, K_diff_data(:, 1), 'r-', 'LineWidth', 2.5, 'Marker', '^', 'MarkerSize', 6); hold on;
    %     plot(iter_data_vec, K_diff_data(:, 2), 'b--', 'LineWidth', 2.5, 'Marker', 'd', 'MarkerSize', 6);
    %     title('数据驱动 K1 收敛过程（初始值+差值绝对值）', 'FontSize', 16, 'FontWeight', 'bold');
    %     xlabel('策略迭代轮数', 'FontSize', 14); 
    %     ylabel('增益值/差值绝对值', 'FontSize', 14);
    %     legend('K1(1) - 水平分量', 'K1(2) - 垂直分量', 'FontSize', 12, 'Location', 'best');
    %     grid on;
    %     set(gca, 'FontSize', 12);
    %     xlim([0, length(iter_data_vec)-1]);
    % 
    %     %% 图8：数据驱动 K2收敛过程（差值绝对值）
    %     figure('Position', [450, 450, 800, 600], 'Name', '数据驱动 K2收敛过程');
    %     plot(iter_data_vec, K_diff_data(:, 3), 'r-', 'LineWidth', 2.5, 'Marker', '^', 'MarkerSize', 6); hold on;
    %     plot(iter_data_vec, K_diff_data(:, 4), 'b--', 'LineWidth', 2.5, 'Marker', 'd', 'MarkerSize', 6);
    %     title('数据驱动 K2 收敛过程（初始值+差值绝对值）', 'FontSize', 16, 'FontWeight', 'bold');
    %     xlabel('策略迭代轮数', 'FontSize', 14); 
    %     ylabel('增益值/差值绝对值', 'FontSize', 14);
    %     legend('K2(1) - 水平分量', 'K2(2) - 垂直分量', 'FontSize', 12, 'Location', 'best');
    %     grid on;
    %     set(gca, 'FontSize', 12);
    %     xlim([0, length(iter_data_vec)-1]);
    % 
    %     %% 显示收敛信息
    %     fprintf('\n=== 收敛过程分析 ===\n');
    %     fprintf('模型驱动方法:\n');
    %     fprintf('  K1最终差值: [%.6f, %.6f]\n', K_diff_model(end, 1), K_diff_model(end, 2));
    %     fprintf('  K2最终差值: [%.6f, %.6f]\n', K_diff_model(end, 3), K_diff_model(end, 4));
    %     fprintf('数据驱动方法:\n');
    %     fprintf('  K1最终差值: [%.6f, %.6f]\n', K_diff_data(end, 1), K_diff_data(end, 2));
    %     fprintf('  K2最终差值: [%.6f, %.6f]\n', K_diff_data(end, 3), K_diff_data(end, 4));
    % 
    % else
    %     fprintf('警告：无法绘制K收敛图，缺少历史数据\n');
    % end
    
%% K值收敛分析
if ~isempty(K_history_model) && ~isempty(K_history_data)
    
    % =====================================================================
    % 1. 计算相邻迭代步之间的差值范数 (您之前的要求)
    % =====================================================================
    
    % 计算模型驱动的K值差值的范数
    K_diff_norm_model = zeros(size(K_history_model, 1), 2); 
    K_diff_norm_model(1, 1) = norm(K_history_model(1, 1:2));
    K_diff_norm_model(1, 2) = norm(K_history_model(1, 3:4));
    for i = 2:size(K_history_model, 1)
        K1_diff_model = K_history_model(i, 1:2) - K_history_model(i-1, 1:2);
        K2_diff_model = K_history_model(i, 3:4) - K_history_model(i-1, 3:4);
        K_diff_norm_model(i, 1) = norm(K1_diff_model);
        K_diff_norm_model(i, 2) = norm(K2_diff_model);
    end
    
    % 计算数据驱动的K值差值的范数
    K_diff_norm_data = zeros(size(K_history_data, 1), 2);
    K_diff_norm_data(1, 1) = norm(K_history_data(1, 1:2));
    K_diff_norm_data(1, 2) = norm(K_history_data(1, 3:4));
    for i = 2:size(K_history_data, 1)
        % K1_diff_data = K_history_data(i, 1:2) - K_history_data(i-1, 1:2);
        % K2_diff_data = K_history_data(i, 3:4) - K_history_data(i-1, 3:4);
        K1_diff_data = K_history_data(i, 1:2) ;
        K2_diff_data = K_history_data(i, 3:4) ;
        K_diff_norm_data(i, 1) = norm(K1_diff_data);
        K_diff_norm_data(i, 2) = norm(K2_diff_data);
    end
    
    iter_model_vec = 0:size(K_history_model, 1)-1;
    iter_data_vec = 0:size(K_history_data, 1)-1;
    
    % 图1：模型驱动 K1 和 K2 收敛过程（差值范数）
    figure('Position', [100, 100, 800, 600]);
    subplot(2,1,1);
    plot(iter_model_vec, K_diff_norm_model(:, 1), 'r-', 'LineWidth', 2.5, 'Marker', 'o', 'MarkerSize', 6);
    xlabel('Number of Iteration', 'FontSize', 12);
    ylabel('||K1||', 'FontSize', 12);
    legend('$$||K_1^{k} ||$$', 'Interpreter', 'latex');
    hold on;
    subplot(2,1,2);
    plot(iter_model_vec, K_diff_norm_model(:, 2), 'b--', 'LineWidth', 2.5, 'Marker', 's', 'MarkerSize', 6);
    % title('模型驱动 K 收敛过程（初始值范数+差值范数）', 'FontSize', 16, 'FontWeight', 'bold');
    xlabel('Number of Iteration', 'FontSize', 12);
    ylabel('||K2 ||', 'FontSize', 12);
    legend('$$||K_2^{k}||$$', 'Interpreter', 'latex');
    grid on;
    set(gca, 'FontSize', 12);
    xlim([0, length(iter_model_vec)-1]);
    
    % 图2：数据驱动 K1 和 K2 收敛过程（差值范数）
    figure('Position', [100, 100, 800, 800], 'Name', '数据驱动 K 收敛过程 (差值范数)');
    subplot(2,1,1);
    plot(iter_data_vec, K_diff_norm_data(:, 1), 'r-', 'LineWidth', 2.5, 'Marker', '^', 'MarkerSize', 6); 
    xlabel('Number of Iteration', 'FontSize', 12);
    ylabel('||K1||', 'FontSize', 12);
    legend('$$||K_1^{k}||$$', 'Interpreter', 'latex');
    hold on;
    subplot(2,1,2);
    plot(iter_data_vec, K_diff_norm_data(:, 2), 'b--', 'LineWidth', 2.5, 'Marker', 'd', 'MarkerSize', 6);
    % title('数据驱动 K 收敛过程（初始值范数+差值范数）', 'FontSize', 16, 'FontWeight', 'bold');
    xlabel('Number of Iteration', 'FontSize', 12);
    ylabel('||K2||', 'FontSize', 12);
    legend('$$||K_2^{k}||$$', 'Interpreter', 'latex');
    grid on;
    set(gca, 'FontSize', 12);
    xlim([0, length(iter_data_vec)-1]);
    
    % =====================================================================
    % 2. 计算与参考值之间的误差范数 (您的新要求)
    % =====================================================================
    
    % 定义参考值
    K1_pi_mode = [0.28782826, 0.23712314];
    K2_pi_mode = [0.31879481, 0.27710491];
    
    % 计算模型驱动的误差范数
    K_error_norm_model = zeros(size(K_history_model, 1), 2);
    for i = 1:size(K_history_model, 1)
        K_error_norm_model(i, 1) = norm(K_history_model(i, 1:2) - K1_pi_mode);
        K_error_norm_model(i, 2) = norm(K_history_model(i, 3:4) - K2_pi_mode);
    end
    
    % 计算数据驱动的误差范数
    K_error_norm_data = zeros(size(K_history_data, 1), 2);
    for i = 1:size(K_history_data, 1)
        K_error_norm_data(i, 1) = norm(K_history_data(i, 1:2) - K1_pi_mode);
        K_error_norm_data(i, 2) = norm(K_history_data(i, 3:4) - K2_pi_mode);
    end
    
    % 图3：模型驱动 K 与参考值误差范数收敛图
    figure('Position', [100, 100, 800, 600], 'Name', '模型驱动 K 与参考值误差');
    subplot(2,1,1);
    plot(iter_model_vec, K_error_norm_model(:, 1), 'r-', 'LineWidth', 2.5, 'Marker', 'o', 'MarkerSize', 6); hold on;
    subplot(2,1,2);
    plot(iter_model_vec, K_error_norm_model(:, 2), 'b--', 'LineWidth', 2.5, 'Marker', 's', 'MarkerSize', 6);
    title('模型驱动 K 与参考值误差范数收敛图', 'FontSize', 16, 'FontWeight', 'bold');
    xlabel('策略迭代轮数', 'FontSize', 14);
    ylabel('误差范数', 'FontSize', 14);
    legend('||K1 - K1_{ref}||', '||K2 - K2_{ref}||', 'FontSize', 12, 'Location', 'best');
    grid on;
    set(gca, 'FontSize', 12);
    xlim([0, length(iter_model_vec)-1]);
    
    % 图4：数据驱动 K 与参考值误差范数收敛图
    figure('Position', [100, 100, 800, 600], 'Name', '数据驱动 K 与参考值误差');
    subplot(2,1,1);
    plot(iter_data_vec, K_error_norm_data(:, 1), 'r-', 'LineWidth', 2.5, 'Marker', '^', 'MarkerSize', 6);
    xlabel('Number of Iteration', 'FontSize', 12);
    ylabel('||K1 - K1_{ref}||', 'FontSize', 12);
    legend('$$||K_1^{k} - K_1^*||$$', 'Interpreter', 'latex');
    hold on;
    subplot(2,1,2);
    plot(iter_data_vec, K_error_norm_data(:, 2), 'b--', 'LineWidth', 2.5, 'Marker', 'd', 'MarkerSize', 6);
    title('数据驱动 K 与参考值误差范数收敛图', 'FontSize', 16, 'FontWeight', 'bold');
    xlabel('Number of Iteration', 'FontSize', 12);
    ylabel('||K2 - K2_{ref}||', 'FontSize', 12);
    legend('$$||K_2^{k} - K_2^*||$$', 'Interpreter', 'latex');
    grid on;
    set(gca, 'FontSize', 12);
    xlim([0, length(iter_data_vec)-1]);
    
    % =====================================================================
    % 3. 显示收敛信息
    % =====================================================================
    fprintf('\n=== 收敛过程分析 ===\n');
    fprintf('--- 相邻迭代差值范数 ---\n');
    fprintf('模型驱动方法:\n');
    fprintf('  K1最终差值范数: %.6f\n', K_diff_norm_model(end, 1));
    fprintf('  K2最终差值范数: %.6f\n', K_diff_norm_model(end, 2));
    fprintf('数据驱动方法:\n');
    fprintf('  K1最终差值范数: %.6f\n', K_diff_norm_data(end, 1));
    fprintf('  K2最终差值范数: %.6f\n', K_diff_norm_data(end, 2));
    
    fprintf('\n--- 与参考值误差范数 ---\n');
    fprintf('模型驱动方法:\n');
    fprintf('  K1最终误差范数: %.6f\n', K_error_norm_model(end, 1));
    fprintf('  K2最终误差范数: %.6f\n', K_error_norm_model(end, 2));
    fprintf('数据驱动方法:\n');
    fprintf('  K1最终误差范数: %.6f\n', K_error_norm_data(end, 1));
    fprintf('  K2最终误差范数: %.6f\n', K_error_norm_data(end, 2));
    
else
    fprintf('警告：无法绘制K收敛图，缺少历史数据\n');
end



    %% 状态演化性能总结
    fprintf('\n=== 状态演化性能分析 ===\n');
    
    % 计算各种性能指标
    energy_model_h = sum(sum(x1_model.^2));
    energy_model_v = sum(sum(x2_model.^2));
    energy_data_h = sum(sum(x1_data.^2));
    energy_data_v = sum(sum(x2_data.^2));
    
    max_model_h = max(max(abs(x1_model)));
    max_model_v = max(max(abs(x2_model)));
    max_data_h = max(max(abs(x1_data)));
    max_data_v = max(max(abs(x2_data)));
    
    final_model_h = abs(x1_model(end, end));
    final_model_v = abs(x2_model(end, end));
    final_data_h = abs(x1_data(end, end));
    final_data_v = abs(x2_data(end, end));
    
    fprintf('模型驱动方法:\n');
    fprintf('  水平状态能量: %.6f, 最大值: %.6f, 终态值: %.6f\n', ...
            energy_model_h, max_model_h, final_model_h);
    fprintf('  垂直状态能量: %.6f, 最大值: %.6f, 终态值: %.6f\n', ...
            energy_model_v, max_model_v, final_model_v);
    
    fprintf('数据驱动方法:\n');
    fprintf('  水平状态能量: %.6f, 最大值: %.6f, 终态值: %.6f\n', ...
            energy_data_h, max_data_h, final_data_h);
    fprintf('  垂直状态能量: %.6f, 最大值: %.6f, 终态值: %.6f\n', ...
            energy_data_v, max_data_v, final_data_v);
    
    % 计算相对差异
    h_energy_diff = abs(energy_model_h - energy_data_h) / energy_model_h * 100;
    v_energy_diff = abs(energy_model_v - energy_data_v) / energy_model_v * 100;
    
    fprintf('相对差异:\n');
    fprintf('  水平状态能量差异: %.2f%%\n', h_energy_diff);
    fprintf('  垂直状态能量差异: %.2f%%\n', v_energy_diff);
end

%% ================ 主程序开始 ================

close all;
clc;

%% 系统参数定义
% 系统模态1
A1 = [0.7  0.2;   % [A11 A12] - 水平状态更新
      0.1  0.6];  % [A21 A22] - 垂直状态更新

A2 = [0.8  0.1;   % 模态2
      0.2  0.7];

% 控制输入矩阵
B1 = [0.5; 0.4];  % [B1; B2] - 对水平和垂直分量的影响
B2 = [0.4; 0.6];

% 跳变概率矩阵
pi = [0.6 0.4; 0.64 0.36];   % 系统模态跳变概率
tau = [0.5 0.5; 0.4 0.6];    % 控制器模态跳变概率

% 系统参数
nx = 2;  % 状态维数
nu = 1;  % 控制输入维数

% 权重矩阵
Q = diag([1, 1]);
R = 2;

% 迭代参数
epsilon = 1e-5;
max_iter = 100;

fprintf('开始理论正确的2D Roesser模型策略迭代...\n');

%% 方法1：标准模型驱动策略迭代
fprintf('\n=== 标准模型驱动策略迭代 ===\n');
[K_model, P_model, iter_model, ~] = standard_model_based_PI(...
    A1, A2, B1, B2, pi, tau, Q, R, epsilon, max_iter);

fprintf('模型驱动结果 (迭代%d次):\n', iter_model);
fprintf('K1 = [%.8f, %.8f]\n', K_model{1});
fprintf('K2 = [%.8f, %.8f]\n', K_model{2});

%% 方法2：理论匹配的数据驱动策略迭代
fprintf('\n=== 理论匹配的数据驱动策略迭代 ===\n');
data_iter_target = iter_model;
fprintf('将数据驱动迭代次数强制与模型驱动一致 (%d 次)...\n', data_iter_target);
[K_data, iter_data] = theory_matched_data_driven_PI(...
    A1, A2, B1, B2, pi, tau, Q, R, data_iter_target, true);

fprintf('数据驱动结果 (迭代%d次):\n', iter_data);
fprintf('K1 = [%.8f, %.8f]\n', K_data{1});
fprintf('K2 = [%.8f, %.8f]\n', K_data{2});

%% 方法3：直接求解验证（作为ground truth）
fprintf('\n=== 直接数值求解验证 ===\n');
[K_direct, P_direct] = direct_solve_riccati(A1, A2, B1, B2, pi, tau, Q, R);

fprintf('直接求解结果:\n');
fprintf('K1 = [%.8f, %.8f]\n', K_direct{1});
fprintf('K2 = [%.8f, %.8f]\n', K_direct{2});

%% 结果比较
fprintf('\n=== 结果比较 ===\n');
fprintf('模型驱动 vs 直接求解:\n');
fprintf('  K1差异: %.10f\n', norm(K_model{1} - K_direct{1}));
fprintf('  K2差异: %.10f\n', norm(K_model{2} - K_direct{2}));

fprintf('数据驱动 vs 直接求解:\n');
fprintf('  K1差异: %.10f\n', norm(K_data{1} - K_direct{1}));
fprintf('  K2差异: %.10f\n', norm(K_data{2} - K_direct{2}));

fprintf('模型驱动 vs 数据驱动:\n');
fprintf('  K1差异: %.10f\n', norm(K_model{1} - K_data{1}));
fprintf('  K2差异: %.10f\n', norm(K_model{2} - K_data{2}));
fprintf('  总差异: %.10f\n', norm(K_model{1} - K_data{1}) + norm(K_model{2} - K_data{2}));

%% 理论分析
fprintf('\n=== 理论分析 ===\n');
analyze_convergence_theory(A1, A2, B1, B2, pi, tau, Q, R, K_direct, P_direct);

%% 稳定性验证
fprintf('\n=== 稳定性验证 ===\n');
verify_stability(A1, A2, B1, B2, K_model, '模型驱动');
verify_stability(A1, A2, B1, B2, K_data, '数据驱动');
verify_stability(A1, A2, B1, B2, K_direct, '直接求解');

%% 闭环仿真比较
fprintf('\n=== 闭环仿真比较 ===\n');
I_max = 40; J_max = 40;
A = {A1, A2}; B = {B1, B2};

x_grid_model = simulate_closed_loop(A, B, K_model, pi, tau, I_max, J_max);
x_grid_data = simulate_closed_loop(A, B, K_data, pi, tau, I_max, J_max);

% 计算总能量
energy_model = sum(sum(sum(x_grid_model.^2)));
energy_data = sum(sum(sum(x_grid_data.^2)));

fprintf('模型驱动总能量: %.8f\n', energy_model);
fprintf('数据驱动总能量: %.8f\n', energy_data);
fprintf('能量差异: %.8f (%.4f%%)\n', abs(energy_model - energy_data), ...
        100*abs(energy_model - energy_data)/energy_model);

%% 绘图
plot_comprehensive_results(x_grid_model, x_grid_data, K_model, K_data, iter_model, iter_data);
