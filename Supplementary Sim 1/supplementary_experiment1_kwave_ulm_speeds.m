clearvars;
parallel.gpu.enableCUDAForwardCompatibility(true);
try gpuDevice(1); catch, end

param.cfl = 0.3;
param.source_threshold = 1e-6;

medium_skin.sound_speed_compression = 1540;
medium_skin.sound_speed_compression_sim = 100;
medium_skin.sound_speed_shear = 5.0;
medium_skin.density = 923.5;
medium_skin.alpha_coeff_shear = 10;
medium_skin.alpha_power_shear = 2.0;
medium_skin.alpha_coeff_compression = 0.1;
medium_skin.alpha_power_compression = 1.5;
medium_skin.sponge_layers = 20;
medium_skin.sponge_alpha_max = 30;

target.update_Hz = 200;
target.cycle_period = 1 / target.update_Hz;
target.sim_duration = target.cycle_period * 15;

grid_cfg.dx = 1.0e-3;
grid_cfg.Lx = 80e-3;
grid_cfg.Ly = 80e-3;
grid_cfg.Lz_skin = 40e-3;

dt = param.cfl * grid_cfg.dx / medium_skin.sound_speed_compression_sim;

Nx = round(grid_cfg.Lx / grid_cfg.dx);
Ny = round(grid_cfg.Ly / grid_cfg.dx);
Nz = round(grid_cfg.Lz_skin / grid_cfg.dx);
if mod(Nx, 2), Nx = Nx + 1; end
if mod(Ny, 2), Ny = Ny + 1; end
if mod(Nz, 2), Nz = Nz + 1; end

kgrid = kWaveGrid(Nx, grid_cfg.dx, Ny, grid_cfg.dx, Nz, grid_cfg.dx);

Nt = ceil(target.sim_duration / dt);
t_vec = (0:Nt-1) * dt;
kgrid.setTime(length(t_vec), dt);

medium = struct();
medium.sound_speed_compression = medium_skin.sound_speed_compression_sim;
medium.sound_speed_shear = medium_skin.sound_speed_shear;
medium.density = medium_skin.density;
medium.alpha_coeff_compression = medium_skin.alpha_coeff_compression * ones(Nx, Ny, Nz, 'single');
medium.alpha_coeff_shear = medium_skin.alpha_coeff_shear * ones(Nx, Ny, Nz, 'single');

sponge_layers = medium_skin.sponge_layers;
sponge_profile = reshape(medium_skin.sponge_alpha_max * linspace(0, 1, sponge_layers).^2, [1, 1, sponge_layers]);
medium.alpha_coeff_compression(:, :, end-sponge_layers+1:end) = medium.alpha_coeff_compression(:, :, end-sponge_layers+1:end) + sponge_profile;
medium.alpha_coeff_shear(:, :, end-sponge_layers+1:end) = medium.alpha_coeff_shear(:, :, end-sponge_layers+1:end) + sponge_profile;

[X_grid, Y_grid] = ndgrid((0:Nx-1) * grid_cfg.dx, (0:Ny-1) * grid_cfg.dx);
X_grid = X_grid - mean(X_grid(:));
Y_grid = Y_grid - mean(Y_grid(:));

ring_width = 4.25e-3;
source_amp_stress = 1e2;

methods = { ...
    struct('name', 'ULM_L_M0p2', 'type', 'ulm_l', 'length', 0.005,  'mach', 0.2), ...
    struct('name', 'ULM_L_M0p5', 'type', 'ulm_l', 'length', 0.0125, 'mach', 0.5), ...
    struct('name', 'ULM_L_M1p0', 'type', 'ulm_l', 'length', 0.025,  'mach', 1.0), ...
    struct('name', 'ULM_L_M1p2', 'type', 'ulm_l', 'length', 0.03,   'mach', 1.2), ...
    struct('name', 'ULM_L_M2p0', 'type', 'ulm_l', 'length', 0.05,   'mach', 2.0) ...
    };

output_dir = fullfile(fileparts(mfilename('fullpath')), 'Outputs_Supplementary_Experiment1');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

if gpuDeviceCount > 0
    data_cast = 'gpuArray-single';
else
    data_cast = 'single';
end
input_args = {'PMLSize', [20, 20, 10], 'PMLInside', false, 'PMLAlpha', [2, 2, 0], 'DataCast', data_cast, 'PlotPML', false, 'PlotLayout', false};

results = cell(1, numel(methods));

for m = 1:numel(methods)
    method = methods{m};
    source_field_2d = zeros(Nx, Ny, Nt, 'single');

    for i = 1:Nt
        t = t_vec(i);
        phase = mod(t, target.cycle_period) / target.cycle_period;
        focus_x = -method.length / 2 + method.length * phase;
        focus_y = 0;
        dist_sq = (X_grid - focus_x).^2 + (Y_grid - focus_y).^2;
        gaussian_point = exp(-dist_sq / (2 * ring_width^2));
        source_field_2d(:, :, i) = gaussian_point * source_amp_stress;
    end

    s_mask_sum = max(source_field_2d, [], 3);
    source = struct();
    source.s_mask = zeros(Nx, Ny, Nz);
    source.s_mask(:, :, 1) = (s_mask_sum > param.source_threshold);
    source.s_mask = (source.s_mask == 1);
    active_indices_2d = find(source.s_mask(:, :, 1));

    fprintf('    DEBUG: Number of active source points: %d\n', numel(active_indices_2d));

    F_full = reshape(source_field_2d, Nx * Ny, Nt);
    source.szz = -F_full(active_indices_2d, :);
    fprintf('    DEBUG: Source Szz size: [%d, %d]\n', size(source.szz, 1), size(source.szz, 2));

    szz_col1 = source.szz(:, 1);
    szz_colMid = source.szz(:, round(Nt/2));
    diff_src = norm(szz_col1 - szz_colMid);
    fprintf('    DEBUG: Source Szz dynamics check: Norm(Col1 - ColMid) = %g\n', diff_src);
    if diff_src < 1e-6
        warning('      Source Szz appears static! Check source generation loop.');
    else
        fprintf('      Source Szz is dynamic (Good).\n');
    end

    sensor = struct();
    mask_cube = false(Nx, Ny, Nz);
    target_depth_m = 1e-3;
    record_depth_idx = max(2, round(target_depth_m / grid_cfg.dx));
    record_depth_idx = min(record_depth_idx, Nz);

    mask_cube(:, :, record_depth_idx) = true;
    sensor.mask = mask_cube;
    sensor.record = {'s', 'p', 'u'};

    sensor_data = pstdElastic3D(kgrid, medium, source, sensor, input_args{:});

    sensor_data.sxy = gather(sensor_data.sxy);
    sensor_data.sxz = gather(sensor_data.sxz);
    sensor_data.syz = gather(sensor_data.syz);
    sensor_data.uz = gather(sensor_data.uz);

    dims = size(sensor_data.sxz);
    nd = ndims(sensor_data.sxz);

    if nd == 2
        n_points = dims(1);
        nt_recorded = dims(2);

        if n_points ~= Nx * Ny
            error('Unexpected sensor data size: Expected Nx*Ny points for single layer recording.');
        end

        sensor_data.sxy = reshape(sensor_data.sxy, [Nx, Ny, 1, nt_recorded]);
        sensor_data.sxz = reshape(sensor_data.sxz, [Nx, Ny, 1, nt_recorded]);
        sensor_data.syz = reshape(sensor_data.syz, [Nx, Ny, 1, nt_recorded]);
        sensor_data.uz = reshape(sensor_data.uz, [Nx, Ny, 1, nt_recorded]);

    elseif nd == 3
        nt_recorded = dims(3);
        if dims(1) == Nx && dims(2) == Ny
            sensor_data.sxy = reshape(sensor_data.sxy, [Nx, Ny, 1, nt_recorded]);
            sensor_data.sxz = reshape(sensor_data.sxz, [Nx, Ny, 1, nt_recorded]);
            sensor_data.syz = reshape(sensor_data.syz, [Nx, Ny, 1, nt_recorded]);
            sensor_data.uz = reshape(sensor_data.uz, [Nx, Ny, 1, nt_recorded]);
        else
            error('Unexpected 3D sensor data size: [%d, %d, %d]. Expected [%d, %d, Nt].', dims(1), dims(2), dims(3), Nx, Ny);
        end

    elseif nd == 4
        if dims(3) ~= 1
            warning(' recorded more than 1 layer? dims(3)=%d', dims(3));
        end
    else
        error('Unsupported sensor_data.sxz dimensionality: %d', nd);
    end

    tau_xy = sensor_data.sxy;
    tau_xz = sensor_data.sxz;
    tau_yz = sensor_data.syz;
    u_z = sensor_data.uz;

    depth_eval_idx = 1;

    max_idx = min([size(tau_xy, 3), size(tau_xz, 3), size(tau_yz, 3)]);
    if max_idx < 1
        error('Sensor data has no depth layers.');
    end

    tau_xy_eval = squeeze(tau_xy(:, :, depth_eval_idx, :));
    tau_xz_eval = squeeze(tau_xz(:, :, depth_eval_idx, :));
    tau_yz_eval = squeeze(tau_yz(:, :, depth_eval_idx, :));
    uz_eval = squeeze(u_z(:, :, depth_eval_idx, :));

    if ndims(tau_xy_eval) == 2
        tau_xy_eval = reshape(tau_xy_eval, [Nx, Ny, 1]);
        tau_xz_eval = reshape(tau_xz_eval, [Nx, Ny, 1]);
        tau_yz_eval = reshape(tau_yz_eval, [Nx, Ny, 1]);
        uz_eval = reshape(uz_eval, [Nx, Ny, 1]);
    end

    tau_mag_eval = sqrt(tau_xy_eval.^2 + tau_xz_eval.^2 + tau_yz_eval.^2);
    tau_mag_eval(~isfinite(tau_mag_eval)) = 0;

    if ndims(tau_mag_eval) == 2
        tau_mag_eval = reshape(tau_mag_eval, [Nx, Ny, 1]);
    elseif ndims(tau_mag_eval) > 3
        tau_mag_eval = reshape(tau_mag_eval, [size(tau_mag_eval, 1), size(tau_mag_eval, 2), size(tau_mag_eval, 3)]);
    end

    Nt_mag = size(tau_mag_eval, 3);
    if ~isfield(sensor, 'record_start_index')
        sensor.record_start_index = 1;
    end
    t_vec_recorded = ((sensor.record_start_index - 1) + (0:Nt_mag-1)) * dt;
    steady_t_start = max(0, target.sim_duration - target.cycle_period);
    steady_idx = find(t_vec_recorded >= steady_t_start);
    if isempty(steady_idx)
        steady_idx = max(1, Nt_mag - round(target.cycle_period / dt) + 1):Nt_mag;
    end

    tau_rms = sqrt(mean(tau_mag_eval(:, :, steady_idx).^2, 3));
    tau_rms = gather(tau_rms);
    tau_rms(~isfinite(tau_rms)) = 0;

    t_end_eval = min(target.sim_duration, t_vec_recorded(end));
    [~, t_idx_end] = min(abs(t_vec_recorded - t_end_eval));
    n_t_snap = size(tau_mag_eval, 3);
    if isempty(n_t_snap) || n_t_snap < 1
        n_t_snap = 1;
    end
    t_idx_end = max(1, min(double(t_idx_end), double(n_t_snap)));
    t_idx_end = floor(t_idx_end);
    if n_t_snap > 1
        tau_snapshot_end = tau_mag_eval(:, :, t_idx_end);
    else
        tau_snapshot_end = tau_mag_eval(:, :);
    end
    [dtaudx, dtaudy] = gradient(tau_snapshot_end, grid_cfg.dx, grid_cfg.dx);
    grad_mag = sqrt(dtaudx.^2 + dtaudy.^2);
    grad_mag = gather(grad_mag);
    grad_mag(~isfinite(grad_mag)) = 0;

    tau_peak = max(tau_mag_eval, [], 3);
    tau_peak = gather(tau_peak);

    tau_dt = diff(tau_mag_eval, 1, 3) / dt;
    tau_dt_peak = max(abs(tau_dt), [], 3);
    tau_dt_peak = gather(tau_dt_peak);

    mid_x = round(Nx / 2);
    mid_y = round(Ny / 2);
    center_waveform = squeeze(tau_mag_eval(mid_x, mid_y, :));
    center_waveform = gather(center_waveform);

    xt_slice = squeeze(tau_mag_eval(:, mid_y, :));
    xt_slice = gather(xt_slice);

    snapshot_times_rel = [0.25, 0.5, 0.75, 1.0] * target.cycle_period;
    snapshot_indices = zeros(1, 4);

    fprintf('  Extracting snapshots for method %s (First Cycle):\n', method.name);
    for k = 1:4
        t_target = snapshot_times_rel(k);
        if t_target > t_vec_recorded(end), t_target = t_vec_recorded(end); end

        [~, idx] = min(abs(t_vec_recorded - t_target));
        snapshot_indices(k) = idx;

        t_snap = t_vec_recorded(idx);
        phase_snap = mod(t_snap, target.cycle_period) / target.cycle_period;
        focus_x_dbg = -method.length / 2 + method.length * phase_snap;
        fprintf('    Snap %d: t=%.3f ms (Idx %d) | Phase=%.2f | Est. Focus X=%.2f mm\n', ...
            k, t_snap * 1000, idx, phase_snap, focus_x_dbg * 1000);
    end

    tau_xy_snapshots = tau_xy_eval(:, :, snapshot_indices);
    tau_xy_snapshots = gather(tau_xy_snapshots);

    uz_snapshots = uz_eval(:, :, snapshot_indices);
    uz_snapshots = gather(uz_snapshots);

    xt_slice_signed = squeeze(tau_xz_eval(:, mid_y, :));
    xt_slice_signed = gather(xt_slice_signed);

    results{m}.snapshot_indices = snapshot_indices;
    results{m}.name = method.name;
    results{m}.mach = method.mach;
    results{m}.length = method.length;
    results{m}.scan_speed = method.length * target.update_Hz;
    results{m}.tau_rms = tau_rms;
    results{m}.tau_peak = tau_peak;
    results{m}.tau_dt_peak = tau_dt_peak;
    results{m}.center_waveform = center_waveform;
    results{m}.xt_slice = xt_slice;
    results{m}.tau_xy_snapshots = tau_xy_snapshots;
    results{m}.uz_snapshots = uz_snapshots;
    results{m}.xt_slice_signed = xt_slice_signed;
    results{m}.grad_mag = grad_mag;
    results{m}.t_end_val = t_vec_recorded(t_idx_end) * 1000;
    results{m}.t_vec = t_vec_recorded;

    roi_span = 20;
    x_idx = (mid_x - roi_span):(mid_x + roi_span);
    y_idx = (mid_y - roi_span):(mid_y + roi_span);

    steady_duration = 10 * target.cycle_period;
    t_start_steady = target.sim_duration - steady_duration;
    steady_t_idx = find(t_vec_recorded >= t_start_steady);

    tau_roi_steady = tau_mag_eval(x_idx, y_idx, steady_t_idx);
    tau_roi_steady_xy = tau_xy_eval(x_idx, y_idx, steady_t_idx);
    tau_roi_steady_xz = tau_xz_eval(x_idx, y_idx, steady_t_idx);
    tau_roi_steady_yz = tau_yz_eval(x_idx, y_idx, steady_t_idx);

    results{m}.tau_roi_steady = gather(tau_roi_steady);
    results{m}.tau_roi_steady_xy = gather(tau_roi_steady_xy);
    results{m}.tau_roi_steady_xz = gather(tau_roi_steady_xz);
    results{m}.tau_roi_steady_yz = gather(tau_roi_steady_yz);
    results{m}.roi_x_vec = kgrid.x_vec(x_idx);
    results{m}.roi_y_vec = kgrid.y_vec(y_idx);
    results{m}.t_vec_steady = t_vec_recorded(steady_t_idx);

    fprintf('Method %s completed. Mach = %.2f, length = %.2f mm, speed = %.2f m/s\n', ...
        method.name, method.mach, method.length * 1000, method.length * target.update_Hz);
end

all_rms = cellfun(@(x) x.tau_rms, results, 'UniformOutput', false);
Global_Max_RMS = max(cellfun(@(x) max(x(:)), all_rms));
if Global_Max_RMS <= eps('single'), Global_Max_RMS = 1; end

all_grad = cellfun(@(x) x.grad_mag, results, 'UniformOutput', false);
Global_Max_Grad = max(cellfun(@(x) max(x(:)), all_grad));
if Global_Max_Grad <= eps('single'), Global_Max_Grad = 1; end

all_peak = cellfun(@(x) x.tau_peak, results, 'UniformOutput', false);
Global_Max_Peak = max(cellfun(@(x) max(x(:)), all_peak));
if Global_Max_Peak <= eps('single'), Global_Max_Peak = 1; end

all_rate = cellfun(@(x) x.tau_dt_peak, results, 'UniformOutput', false);
Global_Max_Rate = max(cellfun(@(x) max(x(:)), all_rate));
if Global_Max_Rate <= eps('single'), Global_Max_Rate = 1; end

fprintf('Global Max RMS: %g\n', Global_Max_RMS);
fprintf('Global Max Peak: %g\n', Global_Max_Peak);
fprintf('Global Max Gradient: %g\n', Global_Max_Grad);
fprintf('Global Max Rate: %g\n', Global_Max_Rate);

fprintf('Saving simulation data to .mat file...\n');
save_path = fullfile(output_dir, 'supplementary_experiment1_data.mat');

x_vec = kgrid.x_vec;
y_vec = kgrid.y_vec;

save(save_path, 'results', 'x_vec', 'y_vec', 'dt', 'target', 'grid_cfg', '-v7.3');
fprintf('Data saved to %s\n', save_path);
