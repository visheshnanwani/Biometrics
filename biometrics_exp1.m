clc; clear; close all;

%% Step 1: Load and Prepare Fingerprint Image
fp_img = imread('102_6.tif');  % Replace with your fingerprint image file
fp_img = imresize(fp_img, [512 512]);

if ndims(fp_img) == 3
    fp_img = rgb2gray(fp_img);
end

fp_img = medfilt2(fp_img, [3 3]);  % Noise reduction

figure, imshow(fp_img); title('Input Fingerprint');

%% Step 2: Background Removal and Region Segmentation
binary_mask = imbinarize(fp_img, 'adaptive', ...
    'ForegroundPolarity', 'dark', 'Sensitivity', 0.45);
binary_mask = imclose(binary_mask, strel('disk', 3));
binary_mask = imfill(binary_mask, 'holes');

region = fp_img;
region(~binary_mask) = 255;

figure, imshow(region); title('Foreground Segmented');

%% Step 3: Orientation Map Estimation
[orient_map, ~] = compute_orientation(region, 1, 3, 3);
orient_deg = orient_map * (180 / pi);

step = 16;
[h, w] = size(region);
[x_coords, y_coords] = meshgrid(1:step:w, 1:step:h);
dir_sample = orient_map(1:step:end, 1:step:end);
dx = cos(dir_sample);
dy = -sin(dir_sample);  % Adjust for image coordinate system

figure, imshow(region); hold on;
quiver(x_coords, y_coords, dx, dy, 0.5, 'r');
title('Orientation Field');

%% Step 4: Detect Singularities (Cores and Deltas)
sing_map = find_singular_points(orient_map);
[core_row, core_col] = find(sing_map == 1);
[delta_row, delta_col] = find(sing_map == -1);

figure, imshow(region); hold on;
plot(core_col, core_row, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
plot(delta_col, delta_row, 'go', 'MarkerSize', 10, 'LineWidth', 2);
legend('Core', 'Delta');
title('Singular Points');

% Pattern Identification
n_core = numel(core_row);
n_delta = numel(delta_row);

if n_core == 1 && n_delta == 1
    pattern_type = 'Loop';
elseif n_core >= 1 && n_delta >= 2
    pattern_type = 'Whorl';
elseif n_core == 0 && n_delta == 0
    pattern_type = 'Arch';
else
    pattern_type = 'Unclassified';
end

disp(['Fingerprint Classification: ', pattern_type]);

%% Step 5: Extract Minutiae (Level 2)
bw_img = imbinarize(region);
skeleton = bwmorph(~bw_img, 'thin', Inf);

[end_pts, bif_pts] = locate_minutiae(skeleton);

figure, imshow(~skeleton); hold on;
plot(end_pts(:,2), end_pts(:,1), 'ro');
plot(bif_pts(:,2), bif_pts(:,1), 'go');
legend('Endings', 'Bifurcations');
title('Minutiae Detection');

%% ---------- SUPPORTING FUNCTIONS BELOW ----------

function [endings, bifurcations] = locate_minutiae(thin_img)
    [r, c] = size(thin_img);
    endings = [];
    bifurcations = [];

    for row = 2:r-1
        for col = 2:c-1
            if thin_img(row, col)
                win = thin_img(row-1:row+1, col-1:col+1);
                count = sum(win(:)) - 1;

                if count == 1
                    endings = [endings; row, col];
                elseif count >= 3
                    bifurcations = [bifurcations; row, col];
                end
            end
        end
    end
end

function singularities = find_singular_points(orient_field)
    [r, c] = size(orient_field);
    singularities = zeros(r, c);
    ang_deg = orient_field * (180 / pi);

    for y = 2:r-1
        for x = 2:c-1
            block = ang_deg(y-1:y+1, x-1:x+1);
            ring = block([1 2 3 6 9 8 7 4 1]);
            diffs = mod(diff(ring) + 180, 360) - 180;
            sum_diff = sum(diffs);

            if abs(sum_diff - 180) < 30
                singularities(y, x) = 1;   % Core
            elseif abs(sum_diff + 180) < 30
                singularities(y, x) = -1;  % Delta
            end
        end
    end
end

function [orient, rel_map] = compute_orientation(img, grad_sigma, block_sigma, smooth_sigma)
    img = double(img);
    fsize = fix(6 * grad_sigma); if mod(fsize,2)==0, fsize = fsize+1; end
    gauss = fspecial('gaussian', fsize, grad_sigma);
    [gx, gy] = gradient(gauss);
    Ix = filter2(gx, img); Iy = filter2(gy, img);
    Ixx = Ix.^2; Iyy = Iy.^2; Ixy = Ix .* Iy;

    bsize = fix(6 * block_sigma); if mod(bsize,2)==0, bsize = bsize+1; end
    block_gauss = fspecial('gaussian', bsize, block_sigma);
    Ixx = filter2(block_gauss, Ixx);
    Iyy = filter2(block_gauss, Iyy);
    Ixy = 2 * filter2(block_gauss, Ixy);

    denominator = sqrt(Ixy.^2 + (Ixx - Iyy).^2) + eps;
    sin2theta = Ixy ./ denominator;
    cos2theta = (Ixx - Iyy) ./ denominator;

    ssize = fix(6 * smooth_sigma); if mod(ssize,2)==0, ssize = ssize+1; end
    smooth_gauss = fspecial('gaussian', ssize, smooth_sigma);
    sin2theta = filter2(smooth_gauss, sin2theta);
    cos2theta = filter2(smooth_gauss, cos2theta);

    orient = 0.5 * atan2(sin2theta, cos2theta);
    rel_map = denominator ./ max(Ixx + Iyy, eps);
end