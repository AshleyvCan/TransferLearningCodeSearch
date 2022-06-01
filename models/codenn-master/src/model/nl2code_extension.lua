JSON = (loadfile "JSON.lua")() 
local os = require "os"
local PAD = 1
local UNK = 2
local START = 3
local END = 4

 function split_on_space(s)
    tokens = {};
    for match in (s.." "):gmatch("(.-) ") do
        table.insert(tokens, match);
    end
    return tokens
end 

function split_on(s, sep)
    if sep == nil then
       sep = "\t"
    end
    local lines={}
    for chars in string.gmatch(s, "([^"..sep.."]+)") do
       table.insert(lines, chars)
    end
    return lines
 end

local function convert_string_to_tokens(nl)
    s = nl:gsub('[%p%c]', '')
    tokens = split_on_space(s)
    return tokens
end

local function tokenizeNL(nl)
    local nl = nl:match( "^%s*(.-)%s*$" ) 
    return convert_string_to_tokens(nl)
end

function check_key_in_table(table, key_value)
    for key, value in pairs(table) do
      if key == key_value then
        return true
      end
    end
    return false
end

local function tokenizeCode(code_snippet, lang)
    
    local code = code_snippet:match( "^%s*(.-)%s*$" ) --.encode("ascii", "replace")
    local typedCode = nil
    if (lang == "sql") then
        local query = SqlTemplate(code)
        typedCode = query.parseSql()
    elseif (lang == "csharp") then
        typedCode = parseCSharp(code)
    elseif (lang == "python") then
        typedCode = split_on(code:match( "^%s*(.-)%s*$" ), '\\s') --.decode("utf-8").split("\s")
    end
    
    local tokens = {}
    for i, x in ipairs(typedCode) do
        print(x)
        tokens[i] = x
    end
    --print((function() local result = list {} for x in typedCode do x end return result end)())
    --local tokens = (function() local result = {} for x in typedCode do result.append(({x:gsub(" +"," ")})[1]) end return result end)()
    return tokens
end


function get_data_map(filename, vocab, dont_skip, max_code_length, max_nl_length)
    local dataset = {}
    local skipped = 0
    file = io.open(filename, "r")

    for line in file:lines() do

        --local qid, rid, nl, code, wt = split_on(({line:gsub(" +"," ")})[1]) --line:match( "^%s*(.-)%s*$" ))
        local data = split_on(({line:gsub(" +"," ")})[1]) --line:match( "^%s*(.-)%s*$" ))
        qid = data[1]
        rid = data[2]
        nl = data[3]
        code = data[4]
        wt = data[5]

       
        local codeToks = tokenizeCode(code, vocab["lang"])
        local nlToks = tokenizeNL(nl)

        local datasetEntry = {["id"] = rid, ["code"] = code, ["code_sizes"] = table.getn(codeToks), ["code_num"] = {}, ["nl_num"] = {}}
       
        for k, tok in pairs(codeToks) do
            --if (not operator_in(tok, vocab["code_to_num"])) then
            if not check_key_in_table(vocab["code_to_num"], tok) then
                vocab["code_to_num"][tok] = UNK
            end
            table.insert(datasetEntry["code_num"], vocab["code_to_num"][tok])
        end
        table.insert(datasetEntry["nl_num"], vocab["nl_to_num"]["CODE_START"])

        for k, word in pairs(nlToks) do
            --if (not operator_in(word, vocab["nl_to_num"])) then
            if (not check_key_in_table(vocab["nl_to_num"], word)) then    
                vocab["nl_to_num"][word] = UNK
            end
            table.insert(datasetEntry["nl_num"], vocab["nl_to_num"][word])
        end
        table.insert(datasetEntry["nl_num"], vocab["nl_to_num"]["CODE_END"])
        if (dont_skip or ((len(datasetEntry["code_num"]) <= max_code_length) and (len(datasetEntry["nl_num"]) <= max_nl_length))) then
            table.insert(dataset, datasetEntry)
        else
            skipped = (skipped + 1)
        end
    end
    file:close()
    --print(("Total size = " + str(len(dataset))))
    print(("Total skipped = " .. skipped))
    local f = io.open(((((os.getenv("CODENN_WORK") .. "/") .. 'nl2code.tmpfile') .. ".") .. opt.language), "w")
    local contents = JSON:encode(dataset)
    f:write(contents)
    io.close(f)

    return dataset
end

