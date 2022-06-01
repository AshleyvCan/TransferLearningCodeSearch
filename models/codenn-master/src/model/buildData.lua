
JSON = (loadfile "JSON.lua")() 

function new_batch(bs, sl)
	  local batch = {}
    batch = {}
    batch.ids = torch.zeros(bs)
    batch.x = torch.ones(opt.max_code_length, bs)
    batch.mask = torch.zeros(opt.max_code_length, bs)
    batch.fmask = torch.zeros(opt.max_code_length, bs)

    batch.y = torch.ones(sl + 1, bs)
    batch.xsizes = torch.ones(bs)
    batch.maxX = 0
    batch.maxY = 0
	  batch.code= {}
    return batch
end

function get_data(filename, vocab, bs, dont_skip)
	
	local dataFile = io.open(filename, 'r')
	local data = JSON:decode(dataFile:read()) --JSON:decode()
	dataFile:close()
	local count = 0
	local dataset = {}
	dataset.size = #data
	dataset.batches = {}
	dataset.batch_size = bs

	local currBatch = nil

	for i = 1, #data do
		if count % bs == 0 then
			if currBatch ~= nil then
				table.insert(dataset.batches, currBatch)
			end
			currBatch = new_batch(bs, opt.max_nl_length)
			count = 0
		end
		count = count + 1

		currBatch.ids[count] = data[i].id
		currBatch.code[count] = data[i].code
		currBatch.xsizes[count] = data[i].code_sizes

		local apparentXSize = math.min(#data[i].code_num, opt.max_code_length)
		local apparentYSize = math.min(#data[i].nl_num, opt.max_nl_length)

      	if apparentXSize > currBatch.maxX then
      		currBatch.maxX = apparentXSize
		end
      	if (apparentYSize + 1) > currBatch.maxY then
      		currBatch.maxY = apparentYSize + 1
		end

		for j = 1, apparentXSize do
			currBatch.x[j][count] = data[i].code_num[j]
			currBatch.fmask[j][count] = 1
		end
		currBatch.mask[apparentXSize][count] = 1

		for j = 1, apparentYSize do
			currBatch.y[j + 1][count] = data[i].nl_num[j]
		end
	end

	if currBatch ~= nil then
		table.insert(dataset.batches, currBatch)
	end
	print('Total size = ' .. dataset.size)
	dataset.max_code_length = opt.max_code_length
	dataset.max_nl_length = opt.max_nl_length

  return dataset
end

function main()

	local cmd = torch.CmdLine()
	cmd:option('-max_nl_length', 80, 'length')
	cmd:option('-max_code_length', 80, 'length')
    cmd:option('-batch_size', 20, 'length')
    cmd:option('-language', 'python', 'sql or csharp')
	cmd:option('-encoder', 'python.encoder', 'encoder')
	cmd:option('-decoder', 'python.decoder', 'decoder')
	cmd:text()
	opt = cmd:parse(arg)
	local working_dir = os.getenv("CODENN_WORK")

	local vocabFile = io.open(working_dir .. "/version_csn/vocab." .. opt.language, 'r')

	local vocab = JSON:decode(vocabFile:read())
	vocabFile:close()
	torch.save(working_dir .. '/vocab.data.' .. opt.language , vocab)

 	torch.save(working_dir .. '/train.data.' .. opt.language, get_data(working_dir .. '/train.txt.'.. opt.language, vocab, opt.batch_size, false))

 	torch.save(working_dir .. '/valid.data.' .. opt.language, get_data(working_dir .. '/valid.txt.' .. opt.language, vocab, opt.batch_size, false))

 	torch.save(working_dir .. '/dev.data.' .. opt.language, get_data(working_dir .. '/dev.txt.' .. opt.language, vocab, 1, true))

	for i=0, 21 do
		test_data1 = get_data(working_dir .. '/test_batch' .. tostring(i) .. '.txt.' .. opt.language, vocab, 1, true)
		torch.save(working_dir .. '/eval'.. tostring(i) .. '.data.' .. opt.language, test_data1)
	end

end

main()
