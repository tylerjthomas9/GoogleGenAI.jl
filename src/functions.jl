"""
    FunctionDeclaration(name, description, parameters)

Create a function declaration with a Dict-based parameters format.
"""
function FunctionDeclaration(
    name::String, description::Union{String,Nothing}, parameters::Dict{String,Any}
)
    param = FunctionParameter(;
        type=get(parameters, "type", "object"),
        description=get(parameters, "description", nothing),
        properties=get(parameters, "properties", nothing),
        items=get(parameters, "items", nothing),
        required=get(parameters, "required", nothing),
    )
    return FunctionDeclaration(name, description, param)
end

"""
    _function_to_declaration(tool::Function) -> FunctionDeclaration

Build a minimal `FunctionDeclaration` from a plain Julia function, using the
function's name and an empty object parameter schema. For typed schemas, pass
a `FunctionDeclaration` or a `Dict` instead.
"""
function _function_to_declaration(tool::Function)
    return FunctionDeclaration(;
        name=string(nameof(tool)), parameters=FunctionParameter(; type="object")
    )
end

_declaration(tool::Function) = to_api_function_declaration(_function_to_declaration(tool))
_declaration(decl::FunctionDeclaration) = to_api_function_declaration(decl)

function _build_tools(tools, function_declarations=nothing)
    built = Dict{String,Any}[]
    declarations = Dict{String,Any}[]
    for tool in something(tools, ())
        if tool isa Union{Function,FunctionDeclaration}
            push!(declarations, _declaration(tool))
        elseif tool isa AbstractDict
            push!(built, string_key_dict(tool))
        else
            throw(ArgumentError("unsupported tool type: $(typeof(tool))"))
        end
    end
    append!(declarations, _declaration.(something(function_declarations, ())))
    isempty(declarations) ||
        push!(built, Dict{String,Any}("functionDeclarations" => declarations))
    return built
end

function _build_tools(config::GenerateContentConfig)
    return _build_tools(config.tools, config.function_declarations)
end

function _build_tool_config(tool_config::Union{Nothing,ToolConfig})
    tool_config === nothing && return nothing
    fc = tool_config.function_calling_config
    fc === nothing && return nothing
    fc_dict = _compact(
        "mode" => fc.mode, "allowedFunctionNames" => fc.allowed_function_names
    )
    return Dict{String,Any}("functionCallingConfig" => fc_dict)
end

"""
    string_to_symbol_keys(dict::AbstractDict) -> Dict{Symbol, Any}

Convert a dictionary with string keys to one with symbol keys.
This is needed for Julia keyword arguments which expect symbols.
"""
function string_to_symbol_keys(dict::AbstractDict)
    return Dict{Symbol,Any}(Symbol(k) => v for (k, v) in pairs(dict))
end

function _call_function(
    functions::AbstractDict{String,<:Function}, name::AbstractString, args
)
    haskey(functions, name) ||
        return Dict{String,Any}("error" => "Function $name not found")
    result = try
        functions[name](; string_to_symbol_keys(args)...)
    catch e
        return Dict{String,Any}("error" => "Error executing function: $e")
    end
    result === nothing && return Dict{String,Any}("status" => "completed")
    result isa AbstractDict && return result
    return Dict{String,Any}("value" => result)
end

function _function_calls(parts)
    calls = FunctionCall[]
    for part in parts
        fc = get(part, :functionCall, nothing)
        fc === nothing && continue
        args = get(fc, :args, Dict{String,Any}())
        args = if args isa AbstractString
            JSON3.read(args, Dict{String,Any})
        else
            Dict{String,Any}(String(k) => v for (k, v) in pairs(args))
        end
        push!(calls, FunctionCall(String(get(fc, :name, "")), args))
    end
    return calls
end

"""
    execute_parallel_function_calls(
        function_calls::Vector,
        functions::Dict{String, <:Function}
    ) -> Dict{String, Any}

Execute multiple function calls in parallel and collect their results.

# Arguments
- `function_calls::Vector`: Function calls from the model (can be Any or FunctionCall)
- `functions::Dict{String, <:Function}`: Dictionary of available functions

# Returns
- `Dict{String, Any}`: Dictionary mapping function names to their results
"""
function execute_parallel_function_calls(
    function_calls::Vector, functions::Dict{String,<:Function}
)
    return Dict{String,Any}(
        fc.name => _call_function(functions, fc.name, fc.args) for fc in function_calls
    )
end

"""
    add_function_result_to_conversation(conversation, function_name::String, function_result)

Append a function result to a conversation for multi-turn function calling and return the
conversation.
"""
function add_function_result_to_conversation(
    conversation::Vector{<:Dict}, function_name::String, function_result::Any
)
    result_str = if function_result isa String
        function_result
    elseif function_result isa Union{Dict,Vector}
        JSON3.write(function_result)
    else
        string(function_result)
    end

    function_response = Dict(
        :role => "function",
        :parts => [
            Dict(
                :functionResponse => Dict(
                    :name => function_name,
                    :response => Dict(:content => result_str),
                ),
            ),
        ],
    )
    push!(conversation, function_response)
    return conversation
end
