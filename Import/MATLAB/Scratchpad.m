all_values = [];

% Extract values from each element of the structure and concatenate them
for i = 1:length(event)
    all_values = [all_values, event(i).value];
end

% Plot the concatenated values as one line
figure;
plot(all_values);

uval = unique(all_values);

for s = 1:size(uval,2)
    idxes(s) = sum(uval(s)==all_values)
end 