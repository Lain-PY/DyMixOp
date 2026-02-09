% High-order finite difference solver for 2D Burgers equation

% Spatial diff: 4th order Laplacian
% Temporal diff: O(dt^5) due to RK4

function Burgers_2d_solver()
    rng(0);

    % Grid size
    M = 64;
    N = 64;
    n_simu_steps = 200;
    dt = 0.0025;  % maximum 0.003
    dx = 1.0 / M;
    nu = 0.005;
    batch = 1200;
    save_dt = 0.025;

    % Get initial condition from random field
    UV_init = random_coef_fourier_series(M, 4, batch);
    U = UV_init(:,:,:,1);
    V = UV_init(:,:,:,2);

    U_record = U;
    V_record = V;

    for step = 1:n_simu_steps
        [U, V] = update_rk4(U, V, nu, dt, dx);
        
        if sum(isnan(U),'all') + sum(isnan(V),'all') > 0
            fprintf('Divergence');
            break;
        end
        if mod(step, round(save_dt/dt)) == 0
            fprintf('%d\n', step);
            U_record = cat(4, U_record, U);
            V_record = cat(4, V_record, V);
        end
    end

    UV = cat(5, U_record, V_record); %(x,y,batch,time,channel)

    % Plotting
    for j = batch:batch
        fig_save_dir = sprintf('./data_figures/batch_%d/', j);
        if ~exist(fig_save_dir, 'dir')
            mkdir(fig_save_dir);
        end
        for i = 1:5:floor((n_simu_steps*dt)/save_dt)
            postProcess_2x3(UV, M, N, 0, 1, 0, 1, i, fig_save_dir);
        end
    end

    % Save data
    data_save_dir = './';
    uv = permute(UV, [4,3,5,1,2]);
    save(sprintf('%s2dburgers_%dx%dx2x%dx%d_dt%.5f_t[0_%.2f]_nu%.5f.mat', ...
        data_save_dir, floor((n_simu_steps*dt)/save_dt), batch, M, N, dt, n_simu_steps*dt, nu), 'uv');
end

function [dsdx, dsdy] = freq_n_ord_derivative(signal, n, dx, dy)
    [N, M] = size(signal,[1,2]);
    u = 2*pi*fftfreq(N, dx);
    v = 2*pi*fftfreq(M, dy);
    [U, V] = meshgrid(u, v);
    kernel_x = (1i*U).^n;
    kernel_y = (1i*V).^n;
    signal_fft = fft2(signal);
    dsdx = real(ifft2(signal_fft .* kernel_x));
    dsdy = real(ifft2(signal_fft .* kernel_y));
end

function [u_t, v_t] = get_temporal_diff(U, V, nu, dx)
    [u_xx, u_yy] = freq_n_ord_derivative(U, 2, dx, dx);
    [v_xx, v_yy] = freq_n_ord_derivative(V, 2, dx, dx);
    laplace_u = u_xx + u_yy;
    laplace_v = v_xx + v_yy;
    [u_x, u_y] = freq_n_ord_derivative(U, 1, dx, dx);
    [v_x, v_y] = freq_n_ord_derivative(V, 1, dx, dx);

    u_t = nu * laplace_u - U .* u_x - V .* u_y;
    v_t = nu * laplace_v - U .* v_x - V .* v_y;
end

function [U, V] = update_rk4(U0, V0, nu, dt, dx)
    [K1_u, K1_v] = get_temporal_diff(U0, V0, nu, dx);

    U1 = U0 + K1_u * dt/2.0;
    V1 = V0 + K1_v * dt/2.0;
    [K2_u, K2_v] = get_temporal_diff(U1, V1, nu, dx);

    U2 = U0 + K2_u * dt/2.0;
    V2 = V0 + K2_v * dt/2.0;
    [K3_u, K3_v] = get_temporal_diff(U2, V2, nu, dx);

    U3 = U0 + K3_u * dt;
    V3 = V0 + K3_v * dt;
    [K4_u, K4_v] = get_temporal_diff(U3, V3, nu, dx);

    U = U0 + dt*(K1_u+2*K2_u+2*K3_u+K4_u)/6.0;
    V = V0 + dt*(K1_v+2*K2_v+2*K3_v+K4_v)/6.0;
end

function postProcess_2x3(truth, Nx, Ny, xmin, xmax, ymin, ymax, num, fig_save_path)
    cor_x = linspace(xmin, xmax, Nx);
    cor_y = linspace(ymin, ymax, Ny);
    [x_star, y_star] = meshgrid(cor_x, cor_y);
    u_star = squeeze(truth(:, :, 1, num, 1));
    v_star = squeeze(truth(:, :, 1, num, 2));

    figure('Position', [100, 100, 550, 350]);

    subplot(1, 2, 1);
    scatter(x_star(:), y_star(:), 8, u_star(:), 's', 'filled');
    colormap('jet');
    clim([-2.2, 0.8]);
    axis square;
    xlim([xmin, xmax]);
    ylim([ymin, ymax]);
    set(gca, 'XTick', [], 'YTick', []);
    title('u (Ref)');
    colorbar;

    subplot(1, 2, 2);
    scatter(x_star(:), y_star(:), 8, v_star(:), 's', 'filled');
    colormap('jet');
    clim([-1.8, 1.2]);
    axis square;
    xlim([xmin, xmax]);
    ylim([ymin, ymax]);
    set(gca, 'XTick', [], 'YTick', []);
    title('v (Ref.)');
    colorbar;

    if ~isempty(fig_save_path)
        saveas(gcf, sprintf('%s2dburgers_%04d.png', fig_save_path, num));
        close all;
    end
end

function u = random_coef_fourier_series(geo_size, L, batch_size)
    x_cor = linspace(0, 1, geo_size+1);
    x_cor = x_cor(1:end-1);
    [XX, YY] = meshgrid(x_cor, x_cor);
    XX = repmat(XX, [1, 1, batch_size, 2]);
    YY = repmat(YY, [1, 1, batch_size, 2]);

    w = zeros(geo_size, geo_size, batch_size, 2);

    for i = -4:L
        for j = -4:L
            a_cof = randn(1, 1, batch_size, 2);
            b_cof = randn(1, 1, batch_size, 2);
            w = w + a_cof .* sin(2 * pi * (i * XX + j * YY)) + ...
                b_cof .* cos(2 * pi * (i * XX + j * YY));
        end
    end

    w_norm = sqrt(sum(w.^2, 4));
    w_max = max(w_norm,[],[1,2]);
    u = 1.5 * w ./ w_max + (2 * rand(1, 1, batch_size, 2) - 1);
end

function f = fftfreq(n, d)
    if mod(n, 2) == 0
        f = [0:(n/2-1), -n/2:-1] / (d*n);
    else
        f = [0:((n-1)/2), -(n-1)/2:-1] / (d*n);
    end
end
