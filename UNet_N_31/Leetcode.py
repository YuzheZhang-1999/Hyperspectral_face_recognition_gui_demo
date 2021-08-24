matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
m = len(matrix)
n = len(matrix[0])
output = []

Lower_bound_row = 0
Up_bound_row = m
Lower_bound_col = 0
Up_bound_col = n

i = 0
j = 0
while Lower_bound_row < Up_bound_row:
        for j in range(Lower_bound_col, Up_bound_col):  #首行向右压入
            output.append(matrix[i][j])
        Lower_bound_row += 1

        for i in range(Lower_bound_row, Up_bound_row):   #末列向下压入
            output.append(matrix[i][j])
        Up_bound_col -= 1

        for j in range(Lower_bound_col, Up_bound_col):   #末行向左压入
            output.append(matrix[i][Up_bound_col-j-1])
        Up_bound_row -= 1
        j = Up_bound_col-j-1

        for i in range(Lower_bound_row, Up_bound_row):   #首列向上压入
            output.append(matrix[Up_bound_row-i][j])
        Lower_bound_col += 1
        i = Up_bound_col-i-1





























k = 1