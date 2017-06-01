Option Explicit
Option Base 1

Function IsInArray(stringToBeFound As String, arr As Variant) As Boolean
    IsInArray = Not IsError(Application.Match(stringToBeFound, arr, 0))
End Function

Public Function ContantArray()
    ContantArray = Array(2, 13, 17)
End Function