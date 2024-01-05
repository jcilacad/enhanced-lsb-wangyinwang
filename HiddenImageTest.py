# ===========================================================================================================================================
# Low Resolution
# Equals 0: (Equals) Difference =  0.0  == Carrier Image Dimensions - ( 200 ,  249 ) === Hidden Image Dimension - ( 75 ,  83 )
# Positive 1: (Positive) Difference =  1.0  == Carrier Image Dimensions - ( 201 ,  209 ) === Hidden Image Dimension - ( 59 ,  89 )
# Negative 1: (Negative) Difference =  -1.0  == Carrier Image Dimensions - ( 201 ,  215 ) === Hidden Image Dimension - ( 73 ,  74 )
# ===========================================================================================================================================
# Mid Resolution
# Equals 0: (Equals) Difference =  0.0  == Carrier Image Dimensions - ( 600 ,  630 ) === Hidden Image Dimension - ( 210 ,  225 )
# Positive 1: (Positive) Difference =  1.0  == Carrier Image Dimensions - ( 601 ,  601 ) === Hidden Image Dimension - ( 215 ,  210 )
# Negative 1: (Negative) Difference =  -1.0  == Carrier Image Dimensions - ( 601 ,  695 ) === Hidden Image Dimension - ( 228 ,  229 )
# ===========================================================================================================================================
# High Resolution
# Equals 0: (Equals) Difference =  0.0  == Carrier Image Dimensions - ( 1100 ,  1184 ) === Hidden Image Dimension - ( 407 ,  400 )
# Positive 1: (Positive) Difference =  1.0  == Carrier Image Dimensions - ( 1117 ,  1189 ) === Hidden Image Dimension - ( 401 ,  414 )
# Negative 1: (Negative) Difference =  -1.0  == Carrier Image Dimensions - ( 1117 ,  1195 ) === Hidden Image Dimension - ( 404 ,  413 )
# ===========================================================================================================================================

## Low Range Test carrier(200, 300) hidden(50 - 100) ##
## Mid Range Test carrier(600, 700) hidden(200 - 250) ##
## High Range Test carrier(1100, 1200) hidden (400, 450) ##

# ===========================================================================================================================================

for x in range(1100, 1200):
    for y in range(1100, 1200):
        quotient = (x * y * 8) / 8
        for a in range(400, 450):
            for b in range(400, 450):
                difference = quotient - (a * b * 8)
                if difference == 0:
                    print("(Equals) Difference = ", difference, " == Carrier Image Dimensions - (", x, ", ", y,
                          ") === Hidden Image Dimension - (", a, ", ", b, ")")
                elif 5 > difference > 0:
                    print("(Positive) Difference = ", difference, " == Carrier Image Dimensions - (", x, ", ", y,
                          ") === Hidden Image Dimension - (", a, ", ", b, ")")
                elif -5 < difference < 0:
                    print("(Negative) Difference = ", difference, " == Carrier Image Dimensions - (", x, ", ", y,
                          ") === Hidden Image Dimension - (", a, ", ", b, ")")
print("Done")