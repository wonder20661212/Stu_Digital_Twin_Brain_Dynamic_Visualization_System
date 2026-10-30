function run_two_step_csv(input_csv, output_csv, algorithm_dir, lambda_value)
    addpath(algorithm_dir);

    X_time_by_roi = readmatrix(input_csv);
    if isempty(X_time_by_roi) || size(X_time_by_roi, 2) < 2
        error('Input CSV must be timepoints x ROI, with at least two ROI columns.');
    end

    % two_step_CD expects variables x samples.
    X = X_time_by_roi';
    lambda_value = str2double(string(lambda_value));
    if isnan(lambda_value) || lambda_value <= 0
        lambda_value = 2;
    end

    [B_est, ~] = two_step_CD(X, lambda_value);
    writematrix(B_est, output_csv);
end
