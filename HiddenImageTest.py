# ===========================================================================================================================================
# Low Resolution
# Equals 0: (Equals) Difference =  0.0  == Carrier Image Dimensions - ( 200 ,  249 ) === Hidden Image Dimension - ( 75 ,  83 )
# Positive 1: (Positive) Difference =  1.0  == Carrier Image Dimensions - ( 201 ,  209 ) === Hidden Image Dimension - ( 59 ,  89 )
# Negative 1: (Negative) Difference =  -1.0  == Carrier Image Dimensions - ( 201 ,  215 ) === Hidden Image Dimension - ( 73 ,  74 )

# Negative 3: (Negative) Difference = -3.0 == Carrier Image Dimensions - ( 201 , 285 ) === Hidden Image Dimension - ( 77 , 93 )
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

# Final Tests for Extraction failure test

# TODO: Low Resolution
# for x in range(1024, 1200):
#     for y in range(1024, 1200):
#         quotient = (x * y * 8) / 8
#         for a in range(350, 400):
#             for b in range(350, 400):

# (Equals) Difference = 0.0 == Carrier Image Dimensions - ( 1025 ,  1080 ) === Hidden Image Dimension - ( 369 ,  375 )
# (Equals) Difference = 0.0 == Carrier Image Dimensions - ( 1025 ,  1080 ) === Hidden Image Dimension - ( 375 ,  369 )
# (Equals) Difference = 0.0 == Carrier Image Dimensions - ( 1026 ,  1040 ) === Hidden Image Dimension - ( 351 ,  380 )

# (Positive) Difference = 5.0 == Carrier Image Dimensions - ( 1025 ,  1093 ) === Hidden Image Dimension - ( 360 ,  389 )
# (Positive) Difference = 5.0 == Carrier Image Dimensions - ( 1025 ,  1093 ) === Hidden Image Dimension - ( 389 ,  360 )
# (Positive) Difference = 3.0 == Carrier Image Dimensions - ( 1025 ,  1115 ) === Hidden Image Dimension - ( 373 ,  383 )

# (Negative) Difference = -2.0 == Carrier Image Dimensions - ( 1025 ,  1054 ) === Hidden Image Dimension - ( 364 ,  371 )
# (Negative) Difference = -2.0 == Carrier Image Dimensions - ( 1025 ,  1054 ) === Hidden Image Dimension - ( 371 ,  364 )
# (Negative) Difference = -8.0 == Carrier Image Dimensions - ( 1025 ,  1080 ) === Hidden Image Dimension - ( 353 ,  392 )

# (Negative) Difference = -7.0 == Carrier Image Dimensions - ( 1183 ,  1055 ) === Hidden Image Dimension - ( 399 ,  391 )
# (Negative) Difference = -8.0 == Carrier Image Dimensions - ( 1194 ,  1056 ) === Hidden Image Dimension - ( 397 ,  397 )
# (Negative) Difference = -9.0 == Carrier Image Dimensions - ( 1199 ,  1033 ) === Hidden Image Dimension - ( 398 ,  389 )
# (Negative) Difference = -10.0 == Carrier Image Dimensions - ( 1190 ,  1033 ) === Hidden Image Dimension - ( 394 ,  390 )

# TODO: Mid Resolution

# for x in range(2048, 2200):
#     for y in range(2048, 2200):
#         quotient = (x * y * 8) / 8
#         for a in range(700, 800):
#             for b in range(700, 800):

# (Equals) Difference = 0.0 == Carrier Image Dimensions - ( 2080 ,  2166 ) === Hidden Image Dimension - ( 780 ,  722 )
# (Equals) Difference = 0.0 == Carrier Image Dimensions - ( 2080 ,  2167 ) === Hidden Image Dimension - ( 715 ,  788 )
# (Equals) Difference = 0.0 == Carrier Image Dimensions - ( 2080 ,  2167 ) === Hidden Image Dimension - ( 788 ,  715 )

# (Positive) Difference = 5.0 == Carrier Image Dimensions - ( 2081 ,  2085 ) === Hidden Image Dimension - ( 728 ,  745 )
# (Positive) Difference = 3.0 == Carrier Image Dimensions - ( 2081 ,  2115 ) === Hidden Image Dimension - ( 722 ,  762 )
# (Positive) Difference = 1.0 == Carrier Image Dimensions - ( 2081 ,  2121 ) === Hidden Image Dimension - ( 725 ,  761 )

# (Negative) Difference = -1.0 == Carrier Image Dimensions - ( 2081 ,  2159 ) === Hidden Image Dimension - ( 710 ,  791 )
# (Negative) Difference = -5.0 == Carrier Image Dimensions - ( 2081 ,  2171 ) === Hidden Image Dimension - ( 747 ,  756 )
# (Negative) Difference = -8.0 == Carrier Image Dimensions - ( 2081 ,  2168 ) === Hidden Image Dimension - ( 758 ,  744 )

# TODO: High Resolution

# for x in range(4096, 4200):
#     for y in range(4096, 4200):
#         quotient = (x * y * 8) / 8
#         for a in range(1400, 1600):
#             for b in range(1400, 1600):

# (Equals) Difference = 0.0 == Carrier Image Dimensions - ( 4096 ,  4169 ) === Hidden Image Dimension - ( 1516 ,  1408 )
# (Equals) Difference = 0.0 == Carrier Image Dimensions - ( 4097 ,  4176 ) === Hidden Image Dimension - ( 1446 ,  1479 )
# (Equals) Difference = 0.0 == Carrier Image Dimensions - ( 4100 ,  4102 ) === Hidden Image Dimension - ( 1435 ,  1465 )

# (Positive) Difference = 1.0 == Carrier Image Dimensions - ( 4101 ,  4101 ) === Hidden Image Dimension - ( 1465 ,  1435 )
# (Positive) Difference = 4.0 == Carrier Image Dimensions - ( 4100 ,  4121 ) === Hidden Image Dimension - ( 1428 ,  1479 )
# (Positive) Difference = 8.0 == Carrier Image Dimensions - ( 4100 ,  4114 ) === Hidden Image Dimension - ( 1489 ,  1416 )

# (Negative) Difference = -4.0 == Carrier Image Dimensions - ( 4102 ,  4138 ) === Hidden Image Dimension - ( 1424 ,  1490 )
# (Negative) Difference = -6.0 == Carrier Image Dimensions - ( 4102 ,  4139 ) === Hidden Image Dimension - ( 1433 ,  1481 )
# (Negative) Difference = -8.0 == Carrier Image Dimensions - ( 4103 ,  4192 ) === Hidden Image Dimension - ( 1421 ,  1513 )


for x in range(1024, 1200):
    for y in range(1024, 1200):
        quotient = (x * y * 8) / 8
        for a in range(350, 400):
            for b in range(350, 400):

                # Low Resolution
                # quotient = 1048576

                # Mid Resolution
                # quotient = 4194304

                # High Resolution
                # quotient = 16777216

                difference = quotient - (a * b * 8)
                if difference == 0:
                    print("\033[92m(Equals) Difference =\033[0m", difference,
                          "\033[92m== Carrier Image Dimensions -\033[0m (", x, ", ", y,
                          ") \033[92m=== Hidden Image Dimension -\033[0m (", a, ", ", b, ")")
                elif 11 > difference > 0:
                    print("\033[91m(Positive) Difference =\033[0m", difference,
                          "\033[91m== Carrier Image Dimensions -\033[0m (", x, ", ", y,
                          ") \033[91m=== Hidden Image Dimension -\033[0m (", a, ", ", b, ")")
                elif -11 < difference < 0:
                    print("\033[94m(Negative) Difference =\033[0m", difference,
                          "\033[94m== Carrier Image Dimensions -\033[0m (", x, ", ", y,
                          ") \033[94m=== Hidden Image Dimension -\033[0m (", a, ", ", b, ")")

print("\n\nDone")
print("\033[92mGreen - (Zero)\033[0m")
print("\033[91mRed - (Positive)\033[0m")
print("\033[94mBlue - (Negative)\033[0m")
